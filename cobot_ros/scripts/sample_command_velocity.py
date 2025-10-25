#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray

def main():
    # Initialize ROS node
    rospy.init_node('demo_velocity_python', anonymous=True)

    # Create a publisher to the velocity controller command topic
    velocity_pub = rospy.Publisher('/velocity_controller/command', Float64MultiArray, queue_size=10)

    # Set the publishing frequency (Hz)
    rate = rospy.Rate(10)  # 10 Hz

    rospy.loginfo("Starting demo_velocity_python")

    # Record the start time
    start_time = rospy.Time.now()

    # Define how long we want to send velocity commands (3 seconds)
    duration = rospy.Duration(2.0)

    # Run until we hit 3 seconds or shutdown
    while not rospy.is_shutdown() and (rospy.Time.now() - start_time) < duration:
        # Create the message
        velocity_msg = Float64MultiArray()
        # Here, we assume a 6-joint robot; adjust the size or values for your case
        # For example, setting only the base joint (Joint_1) to 0.1, rest to 0
        velocity_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Publish the velocity command
        velocity_pub.publish(velocity_msg)

        # Sleep to maintain the 10 Hz rate
        rate.sleep()

    rospy.loginfo("Stopping demo_velocity_python (3 seconds completed)")

    # Optionally, you may want to send a final zero velocity command to stop the robot:
    velocity_msg = Float64MultiArray(data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    velocity_pub.publish(velocity_msg)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
