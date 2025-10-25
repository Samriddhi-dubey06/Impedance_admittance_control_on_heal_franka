#!/usr/bin/env python3
import rospy
import os
import time

def wait_and_launch():
    rospy.init_node("delayed_launch", anonymous=True)

    rospy.loginfo("Delaying launch of dependent nodes...")
    delay_duration = rospy.get_param("~delay_duration", 26)  # Default delay of 10 seconds

    rospy.loginfo(f"Waiting for {delay_duration} seconds before launching dependent nodes...")
    time.sleep(delay_duration)

    dependent_launch = rospy.get_param("~dependent_launch", "")
    if dependent_launch:
        rospy.loginfo(f"Launching dependent nodes: {dependent_launch}")
        os.system(f"roslaunch {dependent_launch}")
    else:
        rospy.logwarn("No dependent launch file specified.")

    rospy.loginfo("Dependent nodes launched successfully. Exiting.")
    rospy.signal_shutdown("Done")

if __name__ == "__main__":
    try:
        wait_and_launch()
    except rospy.ROSInterruptException:
        pass
