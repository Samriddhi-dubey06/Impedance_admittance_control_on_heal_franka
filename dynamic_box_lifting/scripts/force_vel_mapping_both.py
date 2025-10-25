'''
This code publishes the force bimanually on the object to maintatin the force closure 

'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import rospy

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from franka_msgs.msg import FrankaState

# ---- KDL for HEAL ----
import kdl_parser_py.urdf as kdl_urdf
from PyKDL import ChainJntToJacSolver, JntArray, Jacobian

# ---- Franka KDL wrapper (keep your existing path) ----
sys.path.insert(0, '/home/iitgn-robotics/ds_yash/bimanual_ws/src')
from dynamic_box_lifting.utils.jacobian_franka import FrankaKDL

# ======================= Shared Parameters ==========================
K_VEC = np.array([1.0, 0.2, 1.0, 0.1, 0.1, 0.1], dtype=float)
MAX_JOINT_VEL = 0.1  # rad/s
DEADBAND = 0.01      # threshold on |error|

# ======================= HEAL Config ================================
HEAL_URDF = '/home/iitgn-robotics/Samriddhi_WS/bimanual_ws/src/addverb_heal_description/urdf/robot.urdf'
HEAL_BASE_LINK = 'base_link'
HEAL_TIP_LINK  = 'tool'
HEAL_TOPIC_FT      = '/ft_sensor'
HEAL_TOPIC_JOINTS  = '/heal/joint_states'
HEAL_TOPIC_COMMAND = '/heal/velocity_controller/command'
HEAL_LAMBDA_STAR   = np.array([-7.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)  # Fx target

# ======================= Franka Config ==============================
FR3_URDF  = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/addverb_heal_description/urdf/fr3.urdf"
FR3_BASE  = "fr3_link0"
FR3_TIP   = "fr3_link8"
FR3_JOINTS = [
    "fr3_joint1","fr3_joint2","fr3_joint3",
    "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
]
FR3_TOPIC_STATE   = "/fr3/franka_state_controller/franka_states"
FR3_TOPIC_JOINTS  = "/fr3/joint_states"
FR3_TOPIC_COMMAND = "/fr3/joint_velocity_controller/joint_velocity_command"
FR3_LAMBDA_STAR   = np.array([0.0, -7.0, 0.0, 0.0, 0.0, 0.0], dtype=float)   # Fy target

# ======================= Helpers ===============================
def kdl_jacobian(chain, solver, q):
    """Exact copy of your KDL Jacobian extraction for HEAL."""
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

# ======================= Controllers ==========================
class HealController:
    def __init__(self):
        # State
        self.f_left = np.zeros(6, dtype=float)  # measured wrench
        self.joint_positions = None             # np.array, size = n_joints

        # Parse URDF & KDL init
        if not os.path.isfile(HEAL_URDF):
            rospy.logerr(f'[HEAL] URDF not found: {HEAL_URDF}')
            self.chain = None
            self.jac_solver = None
            self.n = 0
        else:
            ok, tree = kdl_urdf.treeFromFile(HEAL_URDF)
            if not ok:
                rospy.logerr('[HEAL] Failed to parse URDF')
                self.chain = None
                self.jac_solver = None
                self.n = 0
            else:
                self.chain = tree.getChain(HEAL_BASE_LINK, HEAL_TIP_LINK)
                self.jac_solver = ChainJntToJacSolver(self.chain)
                self.n = self.chain.getNrOfJoints()
                rospy.loginfo(f'[HEAL] KDL chain initialized with {self.n} joints')

        # ROS I/O
        rospy.Subscriber(HEAL_TOPIC_FT, WrenchStamped, self.ft_callback, queue_size=1)
        rospy.Subscriber(HEAL_TOPIC_JOINTS, JointState, self.joint_cb, queue_size=1)
        self.pub = rospy.Publisher(HEAL_TOPIC_COMMAND, Float64MultiArray, queue_size=None)

    def ft_callback(self, msg: WrenchStamped):
        # You mapped Z-force into Fx index intentionally; kept as-is.
        f_new = np.hstack(([msg.wrench.force.z, 0, 0], [0, 0, 0]))
        self.f_left = f_new

    def joint_cb(self, msg: JointState):
        if len(msg.position) >= 6:
            self.joint_positions = np.array(msg.position[:6], dtype=float)

    def step(self):
        """One control tick; identical math as your HEAL script."""
        if self.chain is None or self.jac_solver is None:
            return  # KDL not ready

        lam_meas = self.f_left.copy()
        lam_star = HEAL_LAMBDA_STAR.copy()

        if self.joint_positions is None:
            q_dot = np.zeros(self.n, dtype=float)
        else:
            error = abs(lam_star[0] - lam_meas[0])  # Fx-only
            if error < DEADBAND:
                q_dot = np.zeros(self.n, dtype=float)
            else:
                v_c = (0.01 * (lam_star - lam_meas)) / K_VEC
                J = kdl_jacobian(self.chain, self.jac_solver, self.joint_positions)
                if J.shape == (6, self.n) and np.linalg.cond(J) < 1e4:
                    q_dot = np.clip(np.linalg.pinv(J).dot(v_c),
                                    -MAX_JOINT_VEL, MAX_JOINT_VEL)
                else:
                    q_dot = np.zeros(self.n, dtype=float)

        self.pub.publish(Float64MultiArray(data=q_dot.tolist()))


class FrankaController:
    def __init__(self):
        # State
        self.f_left = np.zeros(6, dtype=float)  # O_F_ext_hat_K (base-frame external wrench)
        self.joint_positions = None             # np.array, size = 7
        self.n_joints = 7

        # KDL init via your wrapper
        self.franka_kdl = FrankaKDL(
            urdf_file=FR3_URDF,
            base_link=FR3_BASE,
            tip_link=FR3_TIP,
            joint_names=FR3_JOINTS
        )

        # ROS I/O (keep topics & types)
        rospy.Subscriber(FR3_TOPIC_STATE,  FrankaState, self.franka_state_cb, queue_size=1)
        rospy.Subscriber(FR3_TOPIC_JOINTS, JointState,  self.joint_cb,        queue_size=1)
        self.pub = rospy.Publisher(FR3_TOPIC_COMMAND, Float64MultiArray, queue_size=1)

    def franka_state_cb(self, msg: FrankaState):
        try:
            meas = np.array(list(msg.O_F_ext_hat_K), dtype=float)  # [Fx,Fy,Fz,Tx,Ty,Tz]
            self.f_left = meas  # no filtering
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[FR3] Failed to parse O_F_ext_hat_K: {e}")

    def joint_cb(self, msg: JointState):
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        try:
            self.joint_positions = np.array([name_to_pos[j] for j in FR3_JOINTS], dtype=float)
        except KeyError:
            pass

    def step(self):
        """One control tick; identical math as your FR3 script but Fy-only."""
        lam_meas = self.f_left.copy()
        lam_star = FR3_LAMBDA_STAR.copy()

        q_dot = np.zeros(self.n_joints, dtype=float)

        if self.joint_positions is not None and len(self.joint_positions) == self.n_joints:
            force_error = lam_star - lam_meas 

            # Only control Fy
            force_error[0] = 0.0   # no Fx
            force_error[2] = 0.0   # no Fz
            force_error[3:6] = 0.0 # no torques

            if abs(force_error[1]) >= DEADBAND:
                v_c = (0.01 * force_error) / K_VEC
                try:
                    J = self.franka_kdl.compute_jacobian(self.joint_positions)  # 6x7
                    q_dot = np.clip(np.linalg.pinv(J) @ v_c,
                                    -MAX_JOINT_VEL, MAX_JOINT_VEL)
                except Exception as e:
                    rospy.logwarn_throttle(1.0, f"[FR3] Jacobian/command error: {e}")
                    q_dot = np.zeros(self.n_joints, dtype=float)

        # >>> publish must be OUTSIDE the except <<<
        self.pub.publish(Float64MultiArray(data=q_dot.tolist()))

# ======================= Main ================================
def main():
    rospy.init_node("force_velocity_dual_robot")

    heal = HealController()
    franka = FrankaController()

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        heal.step()
        franka.step()
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
