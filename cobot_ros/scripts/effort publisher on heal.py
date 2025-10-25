 #!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64MultiArray

def publish_effort():
    rospy.init_node('effort_publisher', anonymous=True)
    pub = rospy.Publisher('/effort_controller/command', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(100)  # 100 Hz

    msg = Float64MultiArray()
    
    # Step 1: Send the first effort command with 12 on the base joint
    msg.data = [12, 0, 0, 0, 0, 0]
    pub.publish(msg)
    rospy.sleep(0.1)  # Small delay to ensure the first command is processed

    # Step 2: Now allow modifying the base joint value dynamically
    while not rospy.is_shutdown():
        msg.data = [4, 0, 0, 0, 0, 0]  # Modify this value as needed
        pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    try:
        publish_effort()
    except rospy.ROSInterruptException:
        pass

# #!/usr/bin/env python3
# import rospy
# from std_msgs.msg import Float64MultiArray

# def publish_effort():
#     rospy.init_node('effort_publisher', anonymous=True)
#     pub = rospy.Publisher('/effort_controlle r/command', Float64MultiArray, queue_size=10)
#     rate = rospy.Rate(100)  # 100 Hz

#     msg = Float64MultiArray()
#     msg.data = [12, 0, 0, 0, 0, 0]

#     while not rospy.is_shutdown():
#         pub.publish(msg)
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         publish_effort()
#     except rospy.ROSInterruptException:
#         pass 


