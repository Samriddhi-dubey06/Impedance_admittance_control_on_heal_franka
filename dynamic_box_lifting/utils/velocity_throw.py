#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import TwistStamped

def s(T):
    # min-jerk scalar: 0 -> 1 smoothly with zero velocity/accel at ends
    return 10*T**3 - 15*T**4 + 6*T**5

def main():
    rospy.init_node("minjerk_const_velocity_xyz")

    # params
    topic  = rospy.get_param("~topic", "/object_velocity_throw")
    v_tar  = rospy.get_param("~v_target", 0.5)     # target speed for x,y,z
    Tau    = rospy.get_param("~Tau", 0.7)          # ramp duration (s)
    pub_hz = rospy.get_param("~pub_hz", 200.0)

    pub = rospy.Publisher(topic, TwistStamped, queue_size=50)
    rate = rospy.Rate(pub_hz)

    t0 = rospy.Time.now().to_sec()

    while not rospy.is_shutdown():
        now = rospy.Time.now().to_sec()
        T = (now - t0) / max(Tau, 1e-6)

        # min-jerk ramp to v_tar, then hold
        sc = s(T) if T <= 1.0 else 1.0
        v = sc * v_tar

        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.twist.linear.x = v
        msg.twist.linear.y = v
        msg.twist.linear.z = v
        # angular left at 0
        pub.publish(msg)

        rate.sleep()

if __name__ == "__main__":
    main()
