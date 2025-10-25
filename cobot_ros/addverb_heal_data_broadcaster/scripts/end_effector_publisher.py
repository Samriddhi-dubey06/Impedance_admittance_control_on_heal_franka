#!/usr/bin/env python3
import time
import rospy
import PyKDL as kdl
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromParam

class KDLForwardKinematicsNode:
    def __init__(self):
        # Load the URDF from the parameter server
        if not rospy.has_param("/heal/robot_description"):
            rospy.logerr("Parameter robot_description not set")
            exit(1)
        self.robot = URDF.from_parameter_server("/heal/robot_description")
        success, self.tree = treeFromParam("/heal/robot_description")

        if not success:
            rospy.logerr("Failed to construct KDL tree from URDF")
            exit(1)

        # Define base and end-effector links
        base_link = "base_link"
        tip_link = "tool_ff"

        # Extract KDL chain from tree
        self.chain = self.tree.getChain(base_link, tip_link)
        self.n_joints = self.chain.getNrOfJoints()
        rospy.loginfo("KDL chain successfully created with %d joints", self.n_joints)

        # Create a forward kinematics solver
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)

        # Create a list of joint names (skip fixed joints)
        self.joint_names = []
        for i in range(self.chain.getNrOfSegments()):
            segment = self.chain.getSegment(i)
            joint = segment.getJoint()
            # Only consider movable joints
            if joint.getTypeName() != "None":
                self.joint_names.append(joint.getName())

        # Publisher to publish the computed end-effector pose
        self.pose_pub = rospy.Publisher("ee_pose", PoseStamped, queue_size=10)
        # Subscriber to joint states
        rospy.Subscriber("/heal/joint_states", JointState, self.joint_state_callback)

    def joint_state_callback(self, msg):
        # Create a joint array for the forward kinematics computation
        q = kdl.JntArray(self.n_joints)

        # Fill in joint values using the order from self.joint_names
        for i, joint_name in enumerate(self.joint_names):
            try:
                index = msg.name.index(joint_name)
                q[i] = msg.position[index]
            except ValueError:
                rospy.logwarn("Joint '%s' not found in JointState; defaulting to 0", joint_name)
                q[i] = 0.0

        # Compute the forward kinematics: get the end-effector frame
        end_effector_frame = kdl.Frame()
        if self.fk_solver.JntToCart(q, end_effector_frame) < 0:
            rospy.logerr("Failed to compute forward kinematics")
            return

        # Convert KDL Frame to PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "robotA/base_link"  # the base frame for the kinematic chain

        # Set position
        pose_msg.pose.position.x = end_effector_frame.p[0]
        pose_msg.pose.position.y = end_effector_frame.p[1]
        pose_msg.pose.position.z = end_effector_frame.p[2]

        # Get orientation as quaternion from the rotation matrix
        qx, qy, qz, qw = end_effector_frame.M.GetQuaternion()
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        # Publish the end-effector pose
        self.pose_pub.publish(pose_msg)
        rospy.logdebug("Published end-effector pose: %s", pose_msg)

if __name__ == '__main__':
    rospy.init_node("kdl_forward_kinematics_node")
    node = KDLForwardKinematicsNode()
    rospy.loginfo("KDL Forward Kinematics Node Started.")
    rospy.spin()
