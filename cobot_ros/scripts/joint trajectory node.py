#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def publish_trajectory():
    rospy.init_node('joint_trajectory_publisher', anonymous=True)
    pub = rospy.Publisher('/joint_trajectory_controller/command', JointTrajectory, queue_size=10)
    rate = rospy.Rate(50)  

    # Define the joint names 
    joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4', 'Joint_5', 'Joint_6']

    while not rospy.is_shutdown():
        # Create the JointTrajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names

        # Create a single point in the trajectory
        point = JointTrajectoryPoint()
        point.positions = [10, 0, 0, 0, 0, 0] 
        point.time_from_start = rospy.Duration(1)  

        # Assign the point to the trajectory message
        traj_msg.points.append(point)

        # Set the header timestamp to the current time
        traj_msg.header.stamp = rospy.Time.now()

        # Publish the trajectory message
        pub.publish(traj_msg)
        rate.sleep()

if __name__ == "__main__":
    try:
        publish_trajectory()
    except rospy.ROSInterruptException:
        pass
