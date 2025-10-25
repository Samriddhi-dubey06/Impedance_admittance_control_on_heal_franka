#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger, TriggerResponse, SetBool

class ThrowOrchestrator:
    def __init__(self):
        # Params
        self.Tau = rospy.get_param("~Tau", 1.0)  # ramp duration (s)

        # Example release velocities (you can set as params too)
        self.v_rel_franka_x = rospy.get_param("~v_rel_franka_x", 0.1)
        self.v_rel_heal_y   = rospy.get_param("~v_rel_heal_y",   0.1)
        self.v_rel_z        = rospy.get_param("~v_rel_z",        0.2)

        self.pub_hz = rospy.get_param("~pub_hz", 200.0)

        # Publishers (single topic consumed by both controllers)
        self.pub = rospy.Publisher("/object_velocity_cmd", TwistStamped, queue_size=50)
        
        rospy.wait_for_service("/fr3/set_throw_mode")
        rospy.wait_for_service("/heal/set_throw_mode")
        rospy.wait_for_service("/fr3/release_force_closure")
        rospy.wait_for_service("/heal/release_force_closure")
        rospy.wait_for_service("/fr3/restore_force_closure")
        rospy.wait_for_service("/heal/restore_force_closure")

        # Service proxies to controllers
        self.s_franka_scale = rospy.ServiceProxy("/fr3/set_throw_mode", SetBool)
        self.s_heal_scale   = rospy.ServiceProxy("/heal/set_throw_mode",   SetBool)
        self.s_franka_rel   = rospy.ServiceProxy("/fr3/release_force_closure", Trigger)
        self.s_heal_rel     = rospy.ServiceProxy("/heal/release_force_closure",   Trigger)
        self.s_franka_res   = rospy.ServiceProxy("/fr3/restore_force_closure", Trigger)
        self.s_heal_res     = rospy.ServiceProxy("/heal/restore_force_closure",   Trigger)

        # Orchestrator service
        rospy.Service("~throw_now", Trigger, self._srv_throw_now)

        self.throw_active = False
        self.t0 = None
        self.dt = 1.0 / self.pub_hz
        self.last_pub = 0.0

    @staticmethod
    def s(T):
        return 10*T**3 - 15*T**4 + 6*T**5  # min-jerk scalar

    def _srv_throw_now(self, _):
        if self.throw_active:
            return TriggerResponse(success=False, message="Throw already active")

        # Set lift_scale=1.0 on both controllers
        self.s_franka_scale(True)
        self.s_heal_scale(True)

        self.throw_active = True
        self.t0 = rospy.Time.now().to_sec()
        return TriggerResponse(success=True, message="Throw started")

    def step(self):
        if not self.throw_active:
            return

        now = rospy.Time.now().to_sec()
        if now - self.last_pub < self.dt:
            return

        T = (now - self.t0) / self.Tau
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()

        if T <= 1.0:
            s = self.s(T)
            # Publish per-axis components on the SAME topic
            msg.twist.linear.x = s * self.v_rel_franka_x  # Franka uses x
            msg.twist.linear.y = s * self.v_rel_heal_y    # HEAL uses y
            msg.twist.linear.z = s * self.v_rel_z         # both use z
            self.pub.publish(msg)
        else:
            # stop motion
            msg.twist.linear.x = 0.0
            msg.twist.linear.y = 0.0
            msg.twist.linear.z = 0.0
            self.pub.publish(msg)

            # turn off force-closure
            self.s_franka_rel()
            self.s_heal_rel()

            # restore lift_scale=0.2
            self.s_franka_scale(False)
            self.s_heal_scale(False)

            # (optional) later you can call self.s_franka_res()/self.s_heal_res() to re-enable squeeze
            self.throw_active = False

        self.last_pub = now
