#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import TwistStamped
# ---- KDL for HEAL ----
import kdl_parser_py.urdf as kdl_urdf
from PyKDL import ChainJntToJacSolver, JntArray, Jacobian
import os
import csv



# ======================= Default Config (override via kwargs) =======================
DEFAULTS = dict(
    urdf='/home/iitgn-robotics/Samriddhi_WS/bimanual_ws/src/addverb_heal_description/urdf/robot.urdf',
    base_link='base_link',
    tip_link='tool',
    topic_ft='/ft_sensor',
    topic_joints='/heal/joint_states',
    topic_command='/heal/velocity_controller/command',
    lambda_star=np.array([-7.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    k_vec=np.array([1.0, 0.2, 1.0, 0.1, 0.1, 0.1], dtype=float),
    max_joint_vel=0.1,  # rad/s
    deadband=0.01,       # |error| threshold
    mu_min=0.5,         # conservative friction
    rho=0.8,            # 20% safety margin
    delta=0.3,          # noise bias (N)
    kappa=0.20,         # drop threshold (20%)
    f_n_max=20.0,       # actuator/comfort limit (N)
)


def _kdl_jacobian(chain, solver, q):
    """Extract a 6xN Jacobian from KDL."""
    n = chain.getNrOfJoints()
    ja = JntArray(n)
    for i in range(n):
        ja[i] = q[i]
    J_kdl = Jacobian(n)
    solver.JntToJac(ja, J_kdl)
    J = np.zeros((6, n))
    for r in range(6):
        for c in range(n):
            J[r, c] = J_kdl[r, c]
    return J


class HealController:
    """
    Minimal HEAL velocity controller using KDL and a force error on Fx.
    Construct with defaults or override any field in DEFAULTS via kwargs.
    """

    def __init__(self, **kwargs):
        cfg = {**DEFAULTS, **kwargs}
        self.urdf = cfg['urdf']
        self.base_link = cfg['base_link']
        self.tip_link = cfg['tip_link']
        self.topic_ft = cfg['topic_ft']
        self.topic_joints = cfg['topic_joints']
        self.topic_command = cfg['topic_command']
        self.lambda_star = cfg['lambda_star'].astype(float)
        self.k_vec = cfg['k_vec'].astype(float)
        self.max_joint_vel = float(cfg['max_joint_vel'])
        self.deadband = float(cfg['deadband'])
        self.mu_min  = float(cfg['mu_min'])
        self.rho     = float(cfg['rho'])
        self.delta   = float(cfg['delta'])
        self.kappa   = float(cfg['kappa'])
        self.f_n_max = float(cfg['f_n_max'])
        self.have_wrench = False
        
        # --- ramping / delay / caps ---
        self.margin_delay_s = float(rospy.get_param("~margin_delay_s", 0.35))   # wait after contact
        self.rate_limit_n_per_s = float(rospy.get_param("~rate_limit_n_per_s", 2.0))  # max d|Fy|/s
        self.soft_cap_n = float(rospy.get_param("~soft_cap_n", 12.0))           # temporary cap just after contact
        self.cap_grace_s = float(rospy.get_param("~cap_grace_s", 1.5))          # time to use soft cap

        # state for timing
        self.contact_start_time = None
        self.prev_cmd_time = rospy.Time.now().to_sec()

        
        # ---- Contact gating & thresholds ----
        self.contact_thresh = float(rospy.get_param("~contact_thresh", 2.0))  # |Fy| above this = contact (N)
        self.contact_required_hits = int(rospy.get_param("~contact_hits", 3)) # debounce: consecutive hits
        self.shear_thresh = float(rospy.get_param("~shear_thresh", 1.0))      # require tangential > this (N)

        self.in_contact = False
        self._contact_hits = 0

        # Warm-start the drop-guard memory so it doesn't inflate at t=0
        self.f_n_cmd_prev = (1.0 - self.kappa) * abs(self.lambda_star[0])  # was: abs(self.lambda_star[1])
        
        # === CSV setup (final lam_star + measured FT) ===
        csv_dir = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv/heal_logs"
        os.makedirs(csv_dir, exist_ok=True)

        # final setpoint + FT (one file)
        self.csv_file_final = os.path.join(csv_dir, "heal_final_and_ft.csv")

        if not os.path.isfile(self.csv_file_final):
            with open(self.csv_file_final, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time",
                    "lam_star_Fx", "lam_star_Fy", "lam_star_Fz",
                    "lam_star_Tx", "lam_star_Ty", "lam_star_Tz",
                    "Fx_meas", "Fy_meas", "Fz_meas", "Tx_meas", "Ty_meas", "Tz_meas"
                ])

        # optional: throttle to ~20 Hz
        self.log_hz = 20.0
        self.log_dt = 1.0 / self.log_hz
        self.last_log_t = None


        # State
        self.f_left = np.zeros(6, dtype=float)   # measured wrench
        self.joint_positions = None  
        self.obj_vel_x = 0.0

        # Parse URDF & KDL init
        if not os.path.isfile(self.urdf):
            rospy.logerr(f'[HEAL] URDF not found: {self.urdf}')
            self.chain = None
            self.jac_solver = None
            self.n = 0
        else:
            ok, tree = kdl_urdf.treeFromFile(self.urdf)
            if not ok:
                rospy.logerr('[HEAL] Failed to parse URDF')
                self.chain = None
                self.jac_solver = None
                self.n = 0
            else:
                self.chain = tree.getChain(self.base_link, self.tip_link)
                self.jac_solver = ChainJntToJacSolver(self.chain)
                self.n = self.chain.getNrOfJoints()
                rospy.loginfo(f'[HEAL] KDL chain initialized with {self.n} joints')

        # ROS I/O
        rospy.Subscriber(self.topic_ft, WrenchStamped, self._ft_callback, queue_size=1)
        rospy.Subscriber(self.topic_joints, JointState, self._joint_cb, queue_size=1)
        rospy.Subscriber("/object_velocity_cmd", TwistStamped, self._obj_vel_cb, queue_size=1)
        self.pub = rospy.Publisher(self.topic_command, Float64MultiArray, queue_size=1)

    # -------------------- Callbacks --------------------
    def _ft_callback(self, msg: WrenchStamped):
        # Take full wrench in sensor frame
        fx_s = msg.wrench.force.x
        fy_s = msg.wrench.force.y
        fz_s = msg.wrench.force.z
        tx_s = msg.wrench.torque.x
        ty_s = msg.wrench.torque.y
        tz_s = msg.wrench.torque.z
        # Map sensor Z → robot X, and keep others as-is
        fx = fz_s       # robot X = sensor Z
        fy = fy_s       # robot Y = sensor Y
        fz = fx_s       # robot Z = sensor X (if you need it, otherwise 0.0)
        self.f_left = np.array([fx, fy, fz, tx_s, ty_s, tz_s], dtype=float)
        self.have_wrench = True


    def _joint_cb(self, msg: JointState):
        if len(msg.position) >= 6:
            self.joint_positions = np.array(msg.position[:6], dtype=float)
            
    def _obj_vel_cb(self, msg: TwistStamped):
        """Store only the linear.x component from /object_velocity_cmd"""
        self.obj_vel_x = msg.twist.linear.x
        rospy.logdebug_throttle(1.0, f"[FR3] Received object vx = {self.obj_vel_x:.3f}")


    # -------------------- Control Step -----------------
    def step(self):
        """One control tick; Fx-only control."""
        if self.chain is None or self.jac_solver is None:
            return  # KDL not ready
        if not self.have_wrench:
            return

        lam_meas = self.f_left.copy()
        lam_star = self.lambda_star.copy()
        # ---- Margin rule on normal (Fx) ----
        f_n_base = abs(lam_star[0])
        # measured normal (Fx after mapping) and tangential (Fy, Fz)
        f_n_meas = abs(lam_meas[0])
        f_t_meas = np.hypot(lam_meas[1], lam_meas[2])

        # contact debounce
        if f_n_meas > self.contact_thresh:
            self._contact_hits += 1
        else:
            self._contact_hits = 0
        self.in_contact = (self._contact_hits >= self.contact_required_hits)
        
        t_now = rospy.Time.now().to_sec()
        if self.in_contact and self.contact_start_time is None:
            self.contact_start_time = t_now
        elif not self.in_contact:
            self.contact_start_time = None


        # start with base target
        f_n_des = f_n_base

        if self.in_contact:
            # delay before enabling friction margin (lets signals settle)
            delay_ok = (self.contact_start_time is not None) and ((t_now - self.contact_start_time) >= self.margin_delay_s)

            if delay_ok and (f_t_meas > self.shear_thresh):
                f_n_req = (f_t_meas + self.delta) / (self.mu_min * self.rho)
                f_n_des = max(f_n_des, f_n_req)

            # drop-guard only after contact is confirmed and delay observed
            if delay_ok and (f_n_meas < (1.0 - self.kappa) * self.f_n_cmd_prev):
                f_n_des = max(f_n_des, self.f_n_cmd_prev / (1.0 - self.kappa))


        # saturate & compute increment
        cap_now = self.f_n_max
        if self.in_contact and (self.contact_start_time is not None) and ((t_now - self.contact_start_time) <= self.cap_grace_s):
            cap_now = min(cap_now, self.soft_cap_n)
        f_n_des = min(f_n_des, cap_now)

        delta_f_n = max(0.0, f_n_des - f_n_base)


        # write back into lam_star[1] with original sign
        sgn_n = -1.0 if self.lambda_star[0] < 0.0 else 1.0
        lam_star[0] = sgn_n * (f_n_base + delta_f_n)
        lam_star_final = lam_star.copy() 
        
        # --- rate limit on commanded normal magnitude ---
        dt_cmd = max(t_now - self.prev_cmd_time, 1e-3)
        f_prev = abs(self.f_n_cmd_prev)      # previous commanded magnitude
        f_goal = abs(f_n_base + delta_f_n)   # desired new magnitude

        max_step = self.rate_limit_n_per_s * dt_cmd
        f_limited = f_prev + np.clip(f_goal - f_prev, -max_step, max_step)

        sgn_n = -1.0 if self.lambda_star[0] < 0.0 else 1.0
        lam_star[0] = sgn_n * f_limited

        self.prev_cmd_time = t_now

        
        # === Log final lam_star + measured FT at ~20 Hz ===
        t_now = rospy.Time.now().to_sec()
        if (self.last_log_t is None) or (t_now - self.last_log_t >= self.log_dt):
            with open(self.csv_file_final, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    t_now,
                    lam_star_final[0], lam_star_final[1], lam_star_final[2],
                    lam_star_final[3], lam_star_final[4], lam_star_final[5],
                    self.f_left[0], self.f_left[1], self.f_left[2],
                    0.0, 0.0, 0.0  # torques if you don't have them mapped; else use self.f_left[3:6]
                ])
            self.last_log_t = t_now


        if self.joint_positions is None:
            q_dot = np.zeros(self.n, dtype=float)
        # else:
        #     error = abs(lam_star[0] - lam_meas[0])  # Fx-only
        #     if error < self.deadband:
        #         q_dot = np.zeros(self.n, dtype=float)
        else:
                force_error = lam_star_final - lam_meas
                force_error[1] = 0.0
                force_error[2] = 0.0
                force_error[3:6] = 0.0
                v_force = (0.01 * force_error) / self.k_vec

                
                v_obj = np.zeros(6)
                v_obj[2] = self.obj_vel_x
                
                # copy so you don’t overwrite the raw v_obj
                v_obj_scaled = v_obj.copy()

                # scale only the lifting direction (z, index 2)
                lift_scale = 0.2   # e.g. 20% of original speed
                v_obj_scaled[2] *= lift_scale

                # combine
                v_c =  v_obj_scaled
                v_final = v_force +  v_c


                J = _kdl_jacobian(self.chain, self.jac_solver, self.joint_positions)
                if J.shape == (6, self.n) and np.linalg.cond(J) < 1e4:
                    q_dot = np.clip(np.linalg.pinv(J).dot(v_final),
                                    -self.max_joint_vel, self.max_joint_vel)
                else:
                    q_dot = np.zeros(self.n, dtype=float)

        self.pub.publish(Float64MultiArray(data=q_dot.tolist()))
        self.f_n_cmd_prev = abs(lam_star_final[0])

