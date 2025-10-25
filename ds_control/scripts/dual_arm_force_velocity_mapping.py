#!/usr/bin/env python3
import rospy
import os
import numpy as np
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from franka_msgs.msg import FrankaState
import kdl_parser_py.urdf as kdl_urdf
from PyKDL import ChainJntToJacSolver, JntArray, Jacobian

# Parameters
K_vec = np.array([0.5, 0.5, 0.3, 0.2, 0.2, 0.1])
MAX_JOINT_VEL = 0.1
FILTER_ALPHA = 0.1
STOP_THRESHOLD = 0.05  # N

# State
f_left = np.zeros(6)
f_right = np.zeros(6)
joint_left = None
joint_right = None

# Shared desired force along X-axis
lambda_star = np.array([0.1, 0, 0, 0, 0, 0])

# ---- Callbacks ----
def ft_callback(msg):
    global f_left
    fx = msg.wrench.force.x
    f_new = np.hstack(([fx, 0, 0], [0, 0, 0]))
    f_left[:] = FILTER_ALPHA * f_new + (1 - FILTER_ALPHA) * f_left

def franka_state_cb(msg):
    global f_right
    fx = msg.O_F_ext_hat_K[0]  # X-axis force
    f_new = np.hstack(([fx, 0, 0], [0, 0, 0]))
    f_right[:] = FILTER_ALPHA * f_new + (1 - FILTER_ALPHA) * f_right

def joint_cb_left(msg):
    global joint_left
    if len(msg.position) >= 6:
        joint_left = np.array(msg.position[:6])

def joint_cb_right(msg):
    global joint_right
    if len(msg.position) >= 6:
        joint_right = np.array(msg.position[:6])

# ---- Helpers ----
def load_chain(urdf_path, base_link, ee_link):
    if not os.path.isfile(urdf_path):
        rospy.logerr("URDF not found: %s", urdf_path)
        return None, None
    ok, tree = kdl_urdf.treeFromFile(urdf_path)
    if not ok:
        rospy.logerr("Failed to parse URDF: %s", urdf_path)
        return None, None
    chain = tree.getChain(base_link, ee_link)
    solver = ChainJntToJacSolver(chain)
    return chain, solver

def compute_jacobian(chain, solver, q):
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

# ---- Main ----
def main():
    rospy.init_node("dual_arm_admittance_control")

    # Paths to URDFs (already generated)
    urdf_left = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/addverb_heal_description/urdf/robot.urdf"
    urdf_right = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/addverb_heal_description/urdf/fr3.urdf"

    # Load KDL Chains
    chain_left, solver_left = load_chain(urdf_left, "base_link", "tool")
    chain_right, solver_right = load_chain(urdf_right, "fr3_link0", "fr3_link8")

    if chain_left is None or chain_right is None:
        return

    n_left = chain_left.getNrOfJoints()
    n_right = chain_right.getNrOfJoints()

    # Subscribers
    rospy.Subscriber("/ft_sensor", WrenchStamped, ft_callback)
    rospy.Subscriber("/fr3/franka_state_controller/franka_states", FrankaState, franka_state_cb)
    rospy.Subscriber("/heal/joint_states", JointState, joint_cb_left)
    rospy.Subscriber("/fr3/joint_states", JointState, joint_cb_right)

    # Publishers
    pub_left = rospy.Publisher("/heal/velocity_controller/command", Float64MultiArray, queue_size=1)
    pub_right = rospy.Publisher("/fr3/joint_velocity_controller/joint_velocity_command", Float64MultiArray, queue_size=1)

    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        if joint_left is None or joint_right is None:
            rospy.logwarn_throttle(2.0, "Waiting for joint states...")
            rate.sleep()
            continue

        # === Left Arm ===
        lambda_meas_l = f_left.copy()
        error_l = np.abs(lambda_star[0] - lambda_meas_l[0])
        if error_l < STOP_THRESHOLD:
            pub_left.publish(Float64MultiArray(data=[0.0] * n_left))
        else:
            v_c_l = (lambda_star - lambda_meas_l) / K_vec
            J_l = compute_jacobian(chain_left, solver_left, joint_left)
            if np.linalg.cond(J_l) < 1e4:
                q_dot_l = np.linalg.pinv(J_l).dot(v_c_l)
                q_dot_l = np.clip(q_dot_l, -MAX_JOINT_VEL, MAX_JOINT_VEL)
                pub_left.publish(Float64MultiArray(data=q_dot_l.tolist()))
            else:
                rospy.logwarn_throttle(2.0, "[Left] Jacobian near singularity, skipping")

        # === Right Arm ===
        lambda_meas_r = f_right.copy()
        error_r = np.abs(lambda_star[0] - lambda_meas_r[0])
        if error_r < STOP_THRESHOLD:
            pub_right.publish(Float64MultiArray(data=[0.0] * n_right))
        else:
            v_c_r = (lambda_star - lambda_meas_r) / K_vec
            J_r = compute_jacobian(chain_right, solver_right, joint_right)
            if np.linalg.cond(J_r) < 1e4:
                q_dot_r = np.linalg.pinv(J_r).dot(v_c_r)
                q_dot_r = np.clip(q_dot_r, -MAX_JOINT_VEL, MAX_JOINT_VEL)
                pub_right.publish(Float64MultiArray(data=q_dot_r.tolist()))
            else:
                rospy.logwarn_throttle(2.0, "[Right] Jacobian near singularity, skipping")

        rospy.loginfo_throttle(1.0, "[Force] HEAL: %.3f N | Franka: %.3f N", lambda_meas_l[0], lambda_meas_r[0])
        rate.sleep()

if __name__ == "__main__":
    main()
