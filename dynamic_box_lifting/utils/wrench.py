#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import WrenchStamped

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def transform_wrench(R_Oi, p_Oi, w_i, name=""):
    """Transform a wrench from contact frame i to object frame O"""
    f_i = w_i[:3]
    tau_i = w_i[3:]
    f_O = R_Oi @ f_i
    tau_O = R_Oi @ tau_i + np.cross(p_Oi, f_O)

    rospy.loginfo("[%s] f_i=%s tau_i=%s → f_O=%s tau_O=%s",
                  name,
                  np.round(f_i,3), np.round(tau_i,3),
                  np.round(f_O,3), np.round(tau_O,3))

    return np.hstack((f_O, tau_O))

class WrenchAggregator:
    def __init__(self):
        # Params
        self.topic_franka = rospy.get_param("~franka_wrench_topic", "/fr3/wrench_filtered")
        self.topic_heal   = rospy.get_param("~heal_wrench_topic",   "/ft_sensor")
        self.topic_out    = rospy.get_param("~out_topic",          "/wrench")

        # Known transforms (contact → box center)
        self.R_OF = np.array([
            [0, 0, 1],
            [0,-1, 0],
            [1, 0, 0]
        ])
        self.R_OH = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])

        # Contact points (12 cm = 0.12 m along ±y in box frame O)
        self.p_OF = np.array([0.0, -0.12, 0.0])  # Franka (say -y side)
        self.p_OH = np.array([0.0,  0.12, 0.0])  # HEAL   (say +y side)

        # State
        self.wrench_franka = None
        self.wrench_heal   = None

        # Subscribers
        rospy.Subscriber(self.topic_franka, WrenchStamped, self.cb_franka)
        rospy.Subscriber(self.topic_heal,   WrenchStamped, self.cb_heal)

        # Publisher
        self.pub = rospy.Publisher(self.topic_out, WrenchStamped, queue_size=10)

        rospy.loginfo("WrenchAggregator running: combining %s + %s → %s",
                      self.topic_franka, self.topic_heal, self.topic_out)

    def cb_franka(self, msg):
        self.wrench_franka = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ], dtype=float)

        rospy.loginfo("Received FRANKA wrench: %s", np.round(self.wrench_franka,3))
        self.try_publish(msg.header)

    def cb_heal(self, msg):
        self.wrench_heal = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ], dtype=float)

        rospy.loginfo("Received HEAL wrench: %s", np.round(self.wrench_heal,3))
        self.try_publish(msg.header)

    def try_publish(self, header):
        if self.wrench_franka is None or self.wrench_heal is None:
            rospy.logwarn_throttle(2.0, "Waiting for both wrenches...")
            return

        # Transform to object frame
        w_OF = transform_wrench(self.R_OF, self.p_OF, self.wrench_franka, name="FRANKA")
        w_OH = transform_wrench(self.R_OH, self.p_OH, self.wrench_heal,   name="HEAL")

        w_total = w_OF + w_OH

        rospy.loginfo("Combined Wrench: force=%s torque=%s",
                      np.round(w_total[:3],3), np.round(w_total[3:],3))

        # Publish
        out = WrenchStamped()
        out.header = header
        out.header.frame_id = "box_center"

        out.wrench.force.x  = float(w_total[0])
        out.wrench.force.y  = float(w_total[1])
        out.wrench.force.z  = float(w_total[2])
        out.wrench.torque.x = float(w_total[3])
        out.wrench.torque.y = float(w_total[4])
        out.wrench.torque.z = float(w_total[5])

        self.pub.publish(out)

def main():
    rospy.init_node("wrench_aggregator")
    WrenchAggregator()
    rospy.spin()

if __name__ == "__main__":
    main()
