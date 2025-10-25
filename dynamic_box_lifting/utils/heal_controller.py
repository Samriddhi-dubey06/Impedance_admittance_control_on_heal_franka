#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import rospy

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import TwistStamped
import csv

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
    lambda_star=np.array([-7.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    k_vec=np.array([1.0, 0.2, 1.0, 0.1, 0.1, 0.1], dtype=float),
    max_joint_vel=0.1,  # rad/s
    deadband=0.01       # |error| threshold
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
        
             # === CSV setup ===
        csv_dir = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv"
        os.makedirs(csv_dir, exist_ok=True)
        self.csv_file = os.path.join(csv_dir, "heal_force_log_hardcoded.csv")

        # Create file with header if not exists
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time",
                    "lambda_star_Fx","lambda_star_Fy","lambda_star_Fz",
                    "meas_Fx","meas_Fy","meas_Fz"
                ])

        # logging rate control
        self.last_log_t = None
        self.log_dt = 0.05   # log at ~20 Hz



        # State
        self.f_left = np.zeros(6, dtype=float)   # measured wrench
        self.joint_positions = None  
        self.obj_vel_x = 0.0# np.array, size = n_joints

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
        # Intentional mapping: use Z force as Fx input
        f_new = np.hstack(([msg.wrench.force.z, 0, 0], [0, 0, 0]))
        self.f_left = f_new

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

        lam_meas = self.f_left.copy()
        lam_star = self.lambda_star.copy()

             # === Log desired Fx,Fy,Fz and measured Fx,Fy,Fz ===
        t_now = rospy.Time.now().to_sec()
        if (self.last_log_t is None) or (t_now - self.last_log_t >= self.log_dt):
            try:
                with open(self.csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        t_now,
                        lam_star[0], lam_star[1], lam_star[2],   # only forces from lambda_star
                        lam_meas[0], lam_meas[1], lam_meas[2]    # only measured forces
                    ])
                self.last_log_t = t_now
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"[HEAL] CSV log failed: {e}")


        if self.joint_positions is None:
            q_dot = np.zeros(self.n, dtype=float)
        # else:
        #     error = abs(lam_star[0] - lam_meas[0])  # Fx-only
        #     if error < self.deadband:
        #         q_dot = np.zeros(self.n, dtype=float)
        else:
                v_force = (0.01 * (lam_star - lam_meas)) / self.k_vec
                
                v_obj = np.zeros(6)
                v_obj[2] = self.obj_vel_x
                
                # copy so you don’t overwrite the raw v_obj
                v_obj_scaled = v_obj.copy()

                # scale only the lifting direction (z, index 2)
                lift_scale = 0.2   # e.g. 20% of original speed
                v_obj_scaled[2] *= lift_scale

                # combine
                v_c =  v_obj_scaled
                # v_c = np.zeros(6)   # [vx, vy, vz, wx, wy, wz]

                # # Fill in what you want (in meters/sec and rad/sec):
                # v_c[0] = 0.0   # linear x
                # v_c[1] = 0.0   # linear y
                # v_c[2] = 0.01  # linear z (lift up slowly at 1 cm/s)

                # v_c[3] = 0.0   # angular x
                # v_c[4] = 0.0   # angular y
                # v_c[5] = 0.0   # angular z
                
                v_final = v_force + v_c


                J = _kdl_jacobian(self.chain, self.jac_solver, self.joint_positions)
                if J.shape == (6, self.n) and np.linalg.cond(J) < 1e4:
                    q_dot = np.clip(np.linalg.pinv(J).dot(v_final),
                                    -self.max_joint_vel, self.max_joint_vel)
                else:
                    q_dot = np.zeros(self.n, dtype=float)

        self.pub.publish(Float64MultiArray(data=q_dot.tolist()))