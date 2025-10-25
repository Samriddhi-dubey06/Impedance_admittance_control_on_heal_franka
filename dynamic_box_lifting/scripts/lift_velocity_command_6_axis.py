'''
This code publishes the velocity on the topic /object_velocity_cmd that is given to the robots to lift the object (without integrating LP)

'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import TwistStamped, WrenchStamped
from std_msgs.msg import Float64MultiArray
import os
import csv


class ImpedanceVelocityCmd:
    def __init__(self):
        # ---- Topics
        self.vel_topic     = rospy.get_param("~vel_topic", "/current_velocity")
        self.wrench_topic  = rospy.get_param("~wrench_topic", "/wrench")
        self.pose_topic    = rospy.get_param("~pose_topic", "/transformed_pos_euler")
        self.cmd_topic     = rospy.get_param("~cmd_topic", "/object_velocity_cmd")
        
        # === CSV setup for v_cmd ===
        csv_dir = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv/csv_without_lp/all_1"
        os.makedirs(csv_dir, exist_ok=True)
        self.csv_file_vcmd = os.path.join(csv_dir, "v_cmd.csv")

        # Create file with header if it doesn't exist
        if not os.path.isfile(self.csv_file_vcmd):
            with open(self.csv_file_vcmd, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "vx", "vy", "vz", "wx", "wy", "wz"])

        # === 20 Hz logger setup ===
        self.log_hz = 20.0
        self.log_dt = 1.0 / self.log_hz
        self.last_log_t = None


        # ---- Params
        d_val = float(rospy.get_param("~d_value", 1.0))
        k_val = float(rospy.get_param("~k_value", 1.0))
        self.D = np.diag([d_val]*6)
        self.K = np.diag([k_val]*6)

        self.x_des = np.array(rospy.get_param("~x_des",[0.12698208664882937, 0.42681034303311693, -1.779359803456577, -0.11205438725470578, 0.09490407221561747, 0.06855580116200022]), dtype=float)
        self.w_des = np.array(rospy.get_param("~w_des", [0,0,0, 0,0,0]), dtype=float)

        self.euler_in_degrees = bool(rospy.get_param("~euler_in_degrees", False))

        # State
        self._xdot = None
        self._x    = None
        self._w    = None

        # Publisher
        self.pub = rospy.Publisher(self.cmd_topic, TwistStamped, queue_size=10)

        # Subscribers
        self.sub_v = rospy.Subscriber(self.vel_topic, TwistStamped, self.cb_vel, queue_size=50)
        self.sub_w = rospy.Subscriber(self.wrench_topic, WrenchStamped, self.cb_wrench, queue_size=50)
        self.sub_p = rospy.Subscriber(self.pose_topic, Float64MultiArray, self.cb_pose, queue_size=50)

        rospy.loginfo("Node started. Listening:\n  %s (vel)\n  %s (wrench)\n  %s (pose)\nPublishing: %s",
                      self.vel_topic, self.wrench_topic, self.pose_topic, self.cmd_topic)

    # --- Callbacks
    def cb_vel(self, msg: TwistStamped):
        rospy.loginfo("Received velocity msg")
        self._xdot = np.array([msg.twist.linear.x,
                               msg.twist.linear.y,
                               msg.twist.linear.z,
                               msg.twist.angular.x,
                               msg.twist.angular.y,
                               msg.twist.angular.z], dtype=float)
        self.try_compute_and_publish(msg.header)

    def cb_wrench(self, msg: WrenchStamped):
        rospy.loginfo("Received wrench msg")
        self._w = np.array([msg.wrench.force.x,
                            msg.wrench.force.y,
                            msg.wrench.force.z,
                            msg.wrench.torque.x,
                            msg.wrench.torque.y,
                            msg.wrench.torque.z], dtype=float)
        self.try_compute_and_publish(msg.header)
        
    def cb_pose(self, msg: Float64MultiArray):
        data = np.array(msg.data, dtype=float).reshape(-1)
        if data.size < 6:
            rospy.logwarn_throttle(1.0, "pose_euler has <6 elements; got %d", data.size)
            return
        self._x = data[:6].copy()
        if self.euler_in_degrees:
            self._x[3:6] = np.deg2rad(self._x[3:6])

        rospy.loginfo("Received pose Float64MultiArray msg (len=%d)", data.size)
        self.try_compute_and_publish(None)

    # --- Core
    def try_compute_and_publish(self, header_for_stamp):
        if self._xdot is None or self._x is None or self._w is None:
            rospy.logwarn("Missing state: xdot=%s x=%s w=%s",
                          self._xdot, self._x, self._w)
            return

        rospy.loginfo("Computing with: xdot=%s  x=%s  w=%s",
                      np.round(self._xdot,3), np.round(self._x,3), np.round(self._w,3))

        # D_inv = np.linalg.inv(self.D)
        # term = self._w - self.w_des - self.K.dot(self.x_des - self._x)
        # v_cmd = self._xdot + D_inv.dot(term)
        
        
        D_inv = np.linalg.inv(self.D)
        term = self._w - self.w_des + self.K.dot(self.x_des - self._x)
        # term =  self.K.dot( self._x - self.x_des)

        v_cmd = self._xdot - D_inv.dot(term)
        
        if self._x[0] >= self.x_des[0]:
            v_cmd[:] = 0.0
            rospy.loginfo("Stop: x reached desired value (%.4f)", self._x[0])

        rospy.loginfo("v_cmd = [%.4f %.4f %.4f | %.4f %.4f %.4f]",
                      v_cmd[0], v_cmd[1], v_cmd[2], v_cmd[3], v_cmd[4], v_cmd[5])

        out = TwistStamped()
        if header_for_stamp is not None:
            out.header = header_for_stamp
        else:
            out.header.stamp = rospy.Time.now()
        out.twist.linear.x  = float(v_cmd[0])
        out.twist.linear.y  = float(v_cmd[1])
        out.twist.linear.z  = float(v_cmd[2])
        out.twist.angular.x = float(v_cmd[3])
        out.twist.angular.y = float(v_cmd[4])
        out.twist.angular.z = float(v_cmd[5])
        self.pub.publish(out)
        
        # === Log v_cmd at 20 Hz ===
        t_now = rospy.Time.now().to_sec()
        if (self.last_log_t is None) or (t_now - self.last_log_t >= self.log_dt):
            with open(self.csv_file_vcmd, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    t_now,
                    v_cmd[0], v_cmd[1], v_cmd[2],
                    v_cmd[3], v_cmd[4], v_cmd[5]
                ])
            self.last_log_t = t_now


def main():
    rospy.init_node("impedance_velocity_cmd")
    ImpedanceVelocityCmd()
    rospy.spin()

if __name__ == "__main__":
    main()