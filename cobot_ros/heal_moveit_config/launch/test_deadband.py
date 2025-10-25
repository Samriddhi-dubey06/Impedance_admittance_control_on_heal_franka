#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64MultiArray

def publish_effort():
    rospy.init_node('effort_publisher', anonymous=True)
    pub = rospy.Publisher('/effort_controller/command', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(100)  # 100 Hz

    msg = Float64MultiArray()

    # Step 1: Apply 25 N-m for the first 5 data points
    msg.data = [0, 0, 0, 30, 0, 0]
    for _ in range(40):  # Publish 5 times
        if rospy.is_shutdown():
            return
        pub.publish(msg)
        rate.sleep()

    # Step 2: Reduce the torque to 5 N-m
    msg.data = [0, 0, 0, 8, 0, 0]
    while not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    try:
        publish_effort()
    except rospy.ROSInterruptException:
        pass