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
    max_joint_vel=0.6,  # rad/s
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
        # after self.last_log_t = None
        self.start_time = rospy.Time.now().to_sec()   # <--- ADD
        self.obj_vel = np.zeros(3, dtype=float)       # <--- ADD (store [x,y,z] from /object_velocity_throw)

        # === CSV setup for lam_star ===
        csv_dir = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv/csv_without_lp/all_1"
        os.makedirs(csv_dir, exist_ok=True)
        self.csv_file_lam = os.path.join(csv_dir, "lam_star_heal.csv")

        # Create file with header if needed
        if not os.path.isfile(self.csv_file_lam):
            with open(self.csv_file_lam, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])

        # === 20 Hz logger setup ===
        self.log_hz = 20.0
        self.log_dt = 1.0 / self.log_hz
        self.last_log_t = None


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
        rospy.Subscriber("/object_velocity_throw", TwistStamped, self._obj_vel_cb, queue_size=1)
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
        # <--- REPLACE the whole callback body with this
        self.obj_vel = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ], dtype=float)



    # -------------------- Control Step -----------------
    def step(self):
        """One control tick; Fx-only control."""
        if self.chain is None or self.jac_solver is None:
            return  # KDL not ready

        lam_meas = self.f_left.copy()
        lam_star = self.lambda_star.copy()
        
        # === Log lam_star at 20 Hz ===
        t_now = rospy.Time.now().to_sec()
        if (self.last_log_t is None) or (t_now - self.last_log_t >= self.log_dt):
            with open(self.csv_file_lam, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([t_now, lam_star[0], lam_star[1], lam_star[2], lam_star[3], lam_star[4], lam_star[5]])
            self.last_log_t = t_now


        if self.joint_positions is None:
            q_dot = np.zeros(self.n, dtype=float)
        # else:
        #     error = abs(lam_star[0] - lam_meas[0])  # Fx-only
        #     if error < self.deadband:
        #         q_dot = np.zeros(self.n, dtype=float)
        else:
                v_force = (0.01 * (lam_star - lam_meas)) / self.k_vec
                
                # Build v_c: zero until 7s, then pass through y,z from /object_velocity_throw
                v_c = np.zeros(6)
                if (t_now - self.start_time) >= 4.0:
                    v_c[1] = self.obj_vel[1]   # y component from the topic
                    # v_c[2] = self.obj_vel[2]   # z component from the topic
                    v_c[2] = 0.0
                # (leave v_c[0], v_c[3:6] = 0)

                
                v_final = v_force + v_c


                J = _kdl_jacobian(self.chain, self.jac_solver, self.joint_positions)
                if J.shape == (6, self.n) and np.linalg.cond(J) < 1e4:
                    q_dot = np.clip(np.linalg.pinv(J).dot(v_final),
                                    -self.max_joint_vel, self.max_joint_vel)
                else:
                    q_dot = np.zeros(self.n, dtype=float)

        self.pub.publish(Float64MultiArray(data=q_dot.tolist()))

# import os
# import numpy as np
# import rospy

# from std_msgs.msg import Float64MultiArray
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import WrenchStamped
# from geometry_msgs.msg import TwistStamped
# import csv
# from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
# from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse


# # ---- KDL for HEAL ----
# import kdl_parser_py.urdf as kdl_urdf
# from PyKDL import ChainJntToJacSolver, JntArray, Jacobian


# # ======================= Default Config (override via kwargs) =======================
# DEFAULTS = dict(
#     urdf='/home/iitgn-robotics/Samriddhi_WS/bimanual_ws/src/addverb_heal_description/urdf/robot.urdf',
#     base_link='base_link',
#     tip_link='tool',
#     topic_ft='/ft_sensor',
#     topic_joints='/heal/joint_states',
#     topic_command='/heal/velocity_controller/command',
#     lambda_star=np.array([-7.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
#     k_vec=np.array([1.0, 0.2, 1.0, 0.1, 0.1, 0.1], dtype=float),
#     max_joint_vel=0.1,  # rad/s
#     deadband=0.01       # |error| threshold
# )


# def _kdl_jacobian(chain, solver, q):
#     """Extract a 6xN Jacobian from KDL."""
#     n = chain.getNrOfJoints()
#     ja = JntArray(n)
#     for i in range(n):
#         ja[i] = q[i]
#     J_kdl = Jacobian(n)
#     solver.JntToJac(ja, J_kdl)
#     J = np.zeros((6, n))
#     for r in range(6):
#         for c in range(n):
#             J[r, c] = J_kdl[r, c]
#     return J


# class HealController:
#     """
#     Minimal HEAL velocity controller using KDL and a force error on Fx.
#     Construct with defaults or override any field in DEFAULTS via kwargs.
#     """

#     def __init__(self, **kwargs):
#         cfg = {**DEFAULTS, **kwargs}
#         self.urdf = cfg['urdf']
#         self.base_link = cfg['base_link']
#         self.tip_link = cfg['tip_link']
#         self.topic_ft = cfg['topic_ft']
#         self.topic_joints = cfg['topic_joints']
#         self.topic_command = cfg['topic_command']
#         self.lambda_star = cfg['lambda_star'].astype(float)
#         self.k_vec = cfg['k_vec'].astype(float)
#         self.max_joint_vel = float(cfg['max_joint_vel'])
#         self.deadband = float(cfg['deadband'])
#         self.obj_vel = np.zeros(3, dtype=float)   # NEW
#         self.lift_scale_lift  = rospy.get_param("~lift_scale_lift", 0.2)
#         self.lift_scale_throw = rospy.get_param("~lift_scale_throw", 1.0)
#         self.lift_scale       = self.lift_scale_lift
#         self.lambda_star_baseline = self.lambda_star.copy()

#         ...
#         rospy.Subscriber("/object_velocity_cmd", TwistStamped, self._obj_vel_cb, queue_size=1)

#         # --- Services (absolute names) ---
#         rospy.Service("/heal/set_throw_mode", SetBool, self._srv_set_throw_mode)
#         rospy.Service("/heal/release_force_closure", Trigger, self._srv_release_fc)
#         rospy.Service("/heal/restore_force_closure", Trigger, self._srv_restore_fc)
#         # === CSV setup for lam_star ===
#         csv_dir = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv/csv_without_lp/all_1"
#         os.makedirs(csv_dir, exist_ok=True)
#         self.csv_file_lam = os.path.join(csv_dir, "lam_star_heal.csv")

#         # Create file with header if needed
#         if not os.path.isfile(self.csv_file_lam):
#             with open(self.csv_file_lam, "w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["time", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])

#         # === 20 Hz logger setup ===
#         self.log_hz = 20.0
#         self.log_dt = 1.0 / self.log_hz
#         self.last_log_t = None


#         # State
#         self.f_left = np.zeros(6, dtype=float)   # measured wrench
#         self.joint_positions = None  
#         self.obj_vel_x = 0.0# np.array, size = n_joints

#         # Parse URDF & KDL init
#         if not os.path.isfile(self.urdf):
#             rospy.logerr(f'[HEAL] URDF not found: {self.urdf}')
#             self.chain = None
#             self.jac_solver = None
#             self.n = 0
#         else:
#             ok, tree = kdl_urdf.treeFromFile(self.urdf)
#             if not ok:
#                 rospy.logerr('[HEAL] Failed to parse URDF')
#                 self.chain = None
#                 self.jac_solver = None
#                 self.n = 0
#             else:
#                 self.chain = tree.getChain(self.base_link, self.tip_link)
#                 self.jac_solver = ChainJntToJacSolver(self.chain)
#                 self.n = self.chain.getNrOfJoints()
#                 rospy.loginfo(f'[HEAL] KDL chain initialized with {self.n} joints')

#         # ROS I/O
#         rospy.Subscriber(self.topic_ft, WrenchStamped, self._ft_callback, queue_size=1)
#         rospy.Subscriber(self.topic_joints, JointState, self._joint_cb, queue_size=1)
#         rospy.Subscriber("/object_velocity_cmd", TwistStamped, self._obj_vel_cb, queue_size=1)
#         self.pub = rospy.Publisher(self.topic_command, Float64MultiArray, queue_size=1)

#     # -------------------- Callbacks --------------------
#     def _ft_callback(self, msg: WrenchStamped):
#         # Intentional mapping: use Z force as Fx input
#         f_new = np.hstack(([msg.wrench.force.z, 0, 0], [0, 0, 0]))
#         self.f_left = f_new
        
#     def _obj_vel_cb(self, msg: TwistStamped):
#         self.obj_vel = np.array([msg.twist.linear.x,
#                                  msg.twist.linear.y,
#                                  msg.twist.linear.z], dtype=float)

#     def _srv_set_throw_mode(self, req: SetBoolRequest):
#         self.lift_scale = self.lift_scale_throw if req.data else self.lift_scale_lift
#         return SetBoolResponse(success=True, message=f"lift_scale={self.lift_scale:.3f}")

#     def _srv_release_fc(self, _req: TriggerRequest):
#         self.lambda_star = np.zeros(6)
#         return TriggerResponse(success=True, message="lambda_star set to zero")

#     def _srv_restore_fc(self, _req: TriggerRequest):
#         self.lambda_star = self.lambda_star_baseline.copy()
#         return TriggerResponse(success=True, message="lambda_star restored")


#     def _joint_cb(self, msg: JointState):
#         if len(msg.position) >= 6:
#             self.joint_positions = np.array(msg.position[:6], dtype=float)
            
#     # def _obj_vel_cb(self, msg: TwistStamped):
#     #     """Store only the linear.x component from /object_velocity_cmd"""
#     #     self.obj_vel_x = msg.twist.linear.x
#     #     rospy.logdebug_throttle(1.0, f"[FR3] Received object vx = {self.obj_vel_x:.3f}")


#     # -------------------- Control Step -----------------
#     def step(self):
#         """One control tick; Fx-only control."""
#         if self.chain is None or self.jac_solver is None:
#             return  # KDL not ready

#         lam_meas = self.f_left.copy()
#         lam_star = self.lambda_star.copy()
        
#         # === Log lam_star at 20 Hz ===
#         t_now = rospy.Time.now().to_sec()
#         if (self.last_log_t is None) or (t_now - self.last_log_t >= self.log_dt):
#             with open(self.csv_file_lam, "a", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([t_now, lam_star[0], lam_star[1], lam_star[2], lam_star[3], lam_star[4], lam_star[5]])
#             self.last_log_t = t_now


#         if self.joint_positions is None:
#             q_dot = np.zeros(self.n, dtype=float)
#         # else:
#         #     error = abs(lam_star[0] - lam_meas[0])  # Fx-only
#         #     if error < self.deadband:
#         #         q_dot = np.zeros(self.n, dtype=float)
#         else:
#                 v_force = (0.01 * (lam_star - lam_meas)) / self.k_vec

#                 # NEW: object velocity uses y and z for HEAL
#                 v_obj = np.zeros(6)
#                 v_obj[1] = self.obj_vel[1]          # y
#                 v_obj[2] = self.obj_vel[2]          # z
#                 v_obj[:3] *= self.lift_scale

#                 v_final = v_force + v_obj
#                 J = _kdl_jacobian(self.chain, self.jac_solver, self.joint_positions)
#                 if J.shape == (6, self.n) and np.linalg.cond(J) < 1e4:
#                     q_dot = np.clip(np.linalg.pinv(J).dot(v_final),
#                                     -self.max_joint_vel, self.max_joint_vel)
#                 else:
#                     q_dot = np.zeros(self.n, dtype=float)

#         self.pub.publish(Float64MultiArray(data=q_dot.tolist()))
