#!/usr/bin/env python3

import rospy
import moveit_commander
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped

class EndEffectorPositionPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('end_effector_position_publisher', anonymous=True)

        # MoveIt! setup
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("heal_arm")  # Replace with your MoveIt group

        # Publisher for end-effector position
        self.ee_position_pub = rospy.Publisher('/end_effector_position', PointStamped, queue_size=10)

        # Subscriber to joint states
        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)

        rospy.loginfo("End-effector position publisher started.")

    def joint_states_callback(self, msg):
        """
        Callback to compute end-effector position using MoveIt! forward kinematics
        """
        try:
            # Get the current end-effector pose
            ee_pose = self.group.get_current_pose().pose

            # Create PointStamped message
            point_msg = PointStamped()
            point_msg.header.stamp = rospy.Time.now()
            point_msg.header.frame_id = "base_link"  # Change to your robot's base frame
            point_msg.point.x = ee_pose.position.x
            point_msg.point.y = ee_pose.position.y
            point_msg.point.z = ee_pose.position.z

            # Publish the end-effector position
            self.ee_position_pub.publish(point_msg)

        except Exception as e:
            rospy.logerr(f"Error computing end-effector position: {e}")

if __name__ == '__main__':
    try:
        moveit_commander.roscpp_initialize([])
        node = EndEffectorPositionPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()
