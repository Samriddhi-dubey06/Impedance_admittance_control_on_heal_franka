#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# === PARAMETERS ===
n_joints = 6
l1, l2, l3 = 0.3, 0.3, 0.1  # example link lengths (customize)
lambda_damping = 0.1
target_pos = np.array([0.2, 0.1, -0.5])  # target heel position

# === STATE ===
current_joint_angles = np.zeros(n_joints)

def joint_state_callback(msg):
    global current_joint_angles
    current_joint_angles = np.array(msg.position[:n_joints])

def forward_kinematics(theta):
    # Simplified FK — replace with your actual model
    x = l1 * np.cos(theta[0]) + l2 * np.cos(theta[0] + theta[1]) + l3 * np.cos(np.sum(theta[:3]))
    y = l1 * np.sin(theta[0]) + l2 * np.sin(theta[0] + theta[1]) + l3 * np.sin(np.sum(theta[:3]))
    z = 0.0  # Assuming a planar model for now — extend for 3D if needed
    return np.array([x, y, z])

def compute_jacobian(theta, eps=1e-5):
    J = np.zeros((3, n_joints))
    fk0 = forward_kinematics(theta)
    for i in range(n_joints):
        dtheta = np.copy(theta)
        dtheta[i] += eps
        fk_d = forward_kinematics(dtheta)
        J[:, i] = (fk_d - fk0) / eps
    return J

def damped_least_squares(J, dx, damping=0.1):
    JT = J.T
    JJ = JT @ J
    I = np.eye(JJ.shape[0])
    inv = np.linalg.inv(JJ + damping**2 * I)
    return inv @ JT @ dx

def main():
    rospy.init_node('heel_dls_ik')

    rospy.Subscriber("/joint_states", JointState, joint_state_callback)
    pub = rospy.Publisher("/velocity_controller/command", Float64MultiArray, queue_size=10)

    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        # Step 1: FK to get heel position
        current_pos = forward_kinematics(current_joint_angles)

        # Step 2: Compute position error
        dx = target_pos - current_pos

        # Step 3: Compute Jacobian
        J = compute_jacobian(current_joint_angles)

        # Step 4: DLS IK to get joint velocities
        joint_velocities = damped_least_squares(J, dx, lambda_damping)

        # Step 5: Publish
        msg = Float64MultiArray(data=joint_velocities.tolist())
        pub.publish(msg)

        rate.sleep()

if __name__ == "__main__":
    main()
