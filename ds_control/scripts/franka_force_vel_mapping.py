#!/usr/bin/env python3
import rospy
import os
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaState
import kdl_parser_py.urdf as kdl_urdf
from PyKDL import ChainJntToJacSolver, JntArray, Jacobian

# === Parameters ===
K_vec = np.array([0.5, 0.5, 0.3, 0.2, 0.2, 0.2])  # Must be length 6
MAX_JOINT_VEL = 0.1
FILTER_ALPHA = 0.1
STOP_THRESHOLD = 0.05  # N
lambda_star = np.array([0.1, 0, 0, 0, 0, 0])  # Desired force along X

# === State ===
f_right = np.zeros(6)
joint_right = None

# === Callbacks ===
def franka_state_cb(msg):
    global f_right
    fx = msg.O_F_ext_hat_K[0]
    f_new = np.hstack(([fx, 0, 0], [0, 0, 0]))
    f_right[:] = FILTER_ALPHA * f_new + (1 - FILTER_ALPHA) * f_right

def joint_cb_right(msg):
    global joint_right
    joint_map = dict(zip(msg.name, msg.position))
    joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
    try:
        joint_right = np.array([joint_map[name] for name in joint_names])
    except KeyError as e:
        rospy.logwarn_throttle(1.0, f"Joint {e} not found in joint states.")


# === Helper ===
def compute_jacobian(chain, solver, q):
    n = chain.getNrOfJoints()
    q = q[:n]
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

# === Main ===
def main():
    rospy.init_node("franka_admittance_control")

    # --- Load URDF ---
    urdf_path = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/addverb_heal_description/urdf/fr3.urdf"
    if not os.path.isfile(urdf_path):
        rospy.logerr("URDF not found at: %s", urdf_path)
        return
    ok, tree = kdl_urdf.treeFromFile(urdf_path)
    if not ok:
        rospy.logerr("Failed to parse URDF")
        return

    base_link = "panda_link0"
    ee_link = "panda_link8"
    chain = tree.getChain(base_link, ee_link)
    solver = ChainJntToJacSolver(chain)
    n = chain.getNrOfJoints()

    # --- ROS Setup ---
    rospy.Subscriber("/fr3/franka_state_controller/franka_states", FrankaState, franka_state_cb)
    rospy.Subscriber("/fr3/joint_states", JointState, joint_cb_right)
    pub = rospy.Publisher("/fr3/joint_velocity_controller/joint_velocity_command",
                          Float64MultiArray, queue_size=1)

    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        if joint_right is None:
            rospy.logwarn_throttle(2.0, "Waiting for joint states...")
            rate.sleep()
            continue

        lambda_meas = f_right.copy()
        error = np.abs(lambda_star[0] - lambda_meas[0])
        if error < STOP_THRESHOLD:
            pub.publish(Float64MultiArray(data=[0.0] * n))
        else:
            v_c = (lambda_star - lambda_meas) / K_vec
            J = compute_jacobian(chain, solver, joint_right)

            if J.size == 0 or J.shape[1] != n:
                rospy.logwarn_throttle(1.0, "Jacobian is empty or wrong shape. Skipping step.")
                rate.sleep()
                continue


            if np.linalg.cond(J) < 1e4:
                q_dot = np.linalg.pinv(J).dot(v_c)
                q_dot = np.clip(q_dot, -MAX_JOINT_VEL, MAX_JOINT_VEL)
                pub.publish(Float64MultiArray(data=q_dot.tolist()))
            else:
                rospy.logwarn_throttle(2.0, "Jacobian near singularity, skipping")

        rospy.loginfo_throttle(1.0, "Franka X Force: %.3f N", lambda_meas[0])
        rate.sleep()

if __name__ == "__main__":
    main()
