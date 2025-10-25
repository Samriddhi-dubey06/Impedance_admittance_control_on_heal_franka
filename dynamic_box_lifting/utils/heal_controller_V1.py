#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import numpy as np
import rospy

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, TwistStamped

# ---- KDL for HEAL ----
import kdl_parser_py.urdf as kdl_urdf
from PyKDL import ChainJntToJacSolver, JntArray, Jacobian


# ======================= Default Config (override via kwargs) =======================
DEFAULTS = dict(
    urdf='/home/iitgn-robotics/Samriddhi_WS/bimanual_ws/src/addverb_heal_description/urdf/robot.urdf',
    base_link='base_link',
    tip_link='tool',
    topic_ft='/ft_sensor',
    topic_joints='/heal/joint_states',
    topic_command='/heal/velocity_controller/command',

    # Preload target on X (Fx). |lambda_star[0]| becomes N0 (minimum normal force).
    lambda_star=np.array([-7.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),

    # Gains / limits
    k_vec=np.array([1.0, 0.2, 1.0, 0.1, 0.1, 0.1], dtype=float),
    max_joint_vel=0.1,   # rad/s
    deadband=0.01,       # |error| threshold for Fx
    min_normal_n = 7.0,
    # Friction-cone anti-slip (X is normal)
    mu=0.4,              # friction coefficient
    nmax=20.0,           # max allowed |normal force| in N
    compression_sign_x=-1.0,  # set +1.0 if compression reads positive Fx
    log_root="/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv/heal_logs",
    enable_logging=True,


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
    HEAL velocity controller (Fx-only normal force control with friction-cone anti-slip).
    Logs to fixed CSV files (overwrites each run):
      - desired_force.csv
      - heal_wrench_values.csv
      - v_force_force_closure.csv
      - v_c_lifting.csv
      - v_final_force_closure_plus_lifting.csv
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
        self.enable_logging = bool(cfg.get('enable_logging', True))
        self.log_root = os.path.expanduser(cfg.get('log_root'))
        # Logging control (rate limit for CSV writes)
        self.last_log_t = None
        self.log_dt = 0.05   # log at ~20 Hz




        # Friction & preload on X
        self.mu = float(cfg['mu'])
        self.nmax = float(cfg['nmax'])
        self.comp_sign_x = float(cfg['compression_sign_x'])
        self.N0 = float(cfg['min_normal_n'])
        rospy.loginfo(f"[HEAL] Using code file: {__file__}")
        rospy.loginfo(f"[HEAL] MIN preload N0 set to {self.N0:.2f} N; mu={self.mu:.3f}; nmax={self.nmax:.2f}; comp_sign_x={self.comp_sign_x:+.0f}")
        if self.N0 > self.nmax:
            rospy.logwarn(f"[HEAL] N0 ({self.N0:.2f}) > nmax ({self.nmax:.2f}); clamping.")
            self.N0 = self.nmax


        # State
        self.f_left = np.zeros(6, dtype=float)
        self.joint_positions = None
        self.obj_vel_x = 0.0

        # === CSV setup (lambda_star and measured wrench) ===
        csv_dir = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv/"  # your path
        os.makedirs(csv_dir, exist_ok=True)
        self.csv_file = os.path.join(csv_dir, "heal_force_log.csv")

        # Create file with header if it doesn't exist
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time",
                    "lam_star_Fx","lam_star_Fy","lam_star_Fz","lam_star_Tx","lam_star_Ty","lam_star_Tz",
                    "meas_Fx","meas_Fy","meas_Fz","meas_Tx","meas_Ty","meas_Tz"
                ])
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
        fz = fx_s       # robot Z = sensor X (if you need it)
        self.f_left = np.array([fx, fy, fz, tx_s, ty_s, tz_s], dtype=float)

    def _joint_cb(self, msg: JointState):
        if len(msg.position) >= 1:
            # keep exactly n joints if available
            n = self.n if self.n > 0 else min(6, len(msg.position))
            self.joint_positions = np.array(msg.position[:n], dtype=float)

    def _obj_vel_cb(self, msg: TwistStamped):
        self.obj_vel_x = msg.twist.linear.x
        rospy.logdebug_throttle(1.0, f"[HEAL] Received object vx = {self.obj_vel_x:.3f}")

    # -------------------- Control Step -----------------
    def step(self):
        """One control tick; Fx-only control; logs lambda*, v_force, v_c, v_final."""
        if self.chain is None or self.jac_solver is None:
            return  # KDL not ready

        lam_meas = self.f_left.copy()
        t = rospy.get_time()

        # --- Anti-slip & minimum normal along X (compression on X) ---
        Fx = float(lam_meas[0])
        Fy = float(lam_meas[1])
        Fz = float(lam_meas[2])

        # --- Compute target normal along X with a hard lower bound at N0 ---
        Ft = math.sqrt(Fy*Fy + Fz*Fz)
        N_req_ideal = 0.0 if self.mu <= 0.0 else Ft / self.mu

        # First clamp up to the minimum preload
        N_tgt = max(self.N0, N_req_ideal)
        # Then cap to nmax
        if N_tgt > self.nmax:
            N_tgt = self.nmax
        # EXTRA GUARD: never allow target to drop below N0 due to any numeric quirks
        if N_tgt < self.N0:
            rospy.logerr_throttle(1.0, f"[HEAL] BUG GUARD: N_tgt={N_tgt:.3f} < N0={self.N0:.3f}; forcing N_tgt=N0")
            N_tgt = self.N0

        # Integer-only command: round magnitude UP (ceiling)
        # e.g. 7.01 -> 8, 7.0 -> 7
        N_int = int(math.ceil(N_tgt))

        # Build desired wrench: only Fx active with the correct compression sign
        lam_star = np.zeros(6, dtype=float)
        lam_star[0] = self.comp_sign_x * float(N_int)  # comp_sign_x = -1 → lam_star[0] = -N_int

        # Prove what you're writing to the CSV each tick (renders every 0.5s)
        rospy.loginfo_throttle(
            0.5,
            f"[HEAL] N0={self.N0:.2f} Ft/mu={(Ft/self.mu) if self.mu>0 else 0:.2f} "
            f"N_tgt={N_tgt:.2f} N_int={N_int} lam_star_x={lam_star[0]:.2f}"
        )

        # === Make sure lambda_star is set BEFORE any CSV logging ===
        self.lambda_star = lam_star.copy()

        # === Single consistent CSV write (rounded command + measured wrench) ===
        t_now = rospy.Time.now().to_sec()
        if (self.last_log_t is None) or (t_now - self.last_log_t >= self.log_dt):
            try:
                # write command (rounded) and measured wrench to heal_force_log.csv
                with open(self.csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        t_now,
                        float(self.lambda_star[0]), float(self.lambda_star[1]), float(self.lambda_star[2]),
                        float(self.lambda_star[3]), float(self.lambda_star[4]), float(self.lambda_star[5]),
                        float(lam_meas[0]), float(lam_meas[1]), float(lam_meas[2]),
                        float(lam_meas[3]), float(lam_meas[4]), float(lam_meas[5]),
                    ])
                self.last_log_t = t_now
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"[HEAL] CSV log failed: {e}")

        # Compute q_dot
        if self.joint_positions is None:
            q_dot = np.zeros(self.n, dtype=float) if self.n > 0 else np.zeros(0, dtype=float)
        else:
            # Force-closure term with Fx deadband
            force_error = lam_star - lam_meas
            # zero out components we don't control
            force_error[1] = 0.0  # Fy
            force_error[2] = 0.0  # Fz
            force_error[3] = 0.0  # Tx
            force_error[4] = 0.0  # Ty
            force_error[5] = 0.0  # Tz
            if abs(force_error[0]) < self.deadband:
                force_error[0] = 0.0
            v_force = (0.005 * force_error) / self.k_vec

            # Lifting/object-velocity term (map obj_vel_x onto z with scaling)
            v_obj = np.zeros(6)
            v_obj[2] = self.obj_vel_x
            v_obj_scaled = v_obj.copy()
            lift_scale = 0.2
            v_obj_scaled[2] *= lift_scale
            v_c = v_obj_scaled

            # Final twist: currently v_force only (enable sum if desired)
            v_final = v_force + v_c
            # v_final = v_force + v_c

            # J and q_dot
            J = _kdl_jacobian(self.chain, self.jac_solver, self.joint_positions)
            if J.shape == (6, self.n) and np.linalg.cond(J) < 1e4:
                q_dot = np.clip(np.linalg.pinv(J).dot(v_final),
                                -self.max_joint_vel, self.max_joint_vel)
            else:
                q_dot = np.zeros(self.n, dtype=float)

        self.pub.publish(Float64MultiArray(data=q_dot.tolist()))
