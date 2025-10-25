#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from tf.transformations import quaternion_matrix

def rotmat_from_quat(qx, qy, qz, qw):
    M = quaternion_matrix([qx, qy, qz, qw])
    return M[:3, :3]

def axis_angle_from_R(R):
    # Returns (axis, angle) with angle in [0, pi], robust to numerical noise
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)

    if angle < 1e-12:
        # Very small rotation
        return np.array([0.0, 0.0, 0.0]), 0.0

    # For general case:
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]
    axis = np.array([rx, ry, rz]) / (2.0 * np.sin(angle))

    # Normalize (safety)
    n = np.linalg.norm(axis)
    if n > 1e-12:
        axis = axis / n
    else:
        axis = np.array([0.0, 0.0, 0.0])

    return axis, angle

class VelocityFromPose:
    def __init__(self):
        # Params / topics
        self.pose_topic   = rospy.get_param("~pose_topic", "/transformed_pos")
        self.twist_topic  = rospy.get_param("~twist_topic", "/current_velocity")

        # State
        self.prev_t      = None
        self.prev_p      = None  # 3x
        self.prev_R      = None  # 3x3

        # I/O
        self.sub = rospy.Subscriber(self.pose_topic, PoseStamped, self.cb, queue_size=50)
        self.pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=50)

        rospy.loginfo("VelocityFromPose: subscribing %s, publishing %s", self.pose_topic, self.twist_topic)

    def cb(self, msg: PoseStamped):
        # Current time (sec)
        t = msg.header.stamp.to_sec()
        # Position (marker_rotated frame coordinates already)
        p = np.array([msg.pose.position.x,
                      msg.pose.position.y,
                      msg.pose.position.z], dtype=float)
        # Orientation → rotation matrix
        qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        R = rotmat_from_quat(qx, qy, qz, qw)

        if self.prev_t is None:
            # First sample: just initialize
            self.prev_t, self.prev_p, self.prev_R = t, p, R
            return

        dt = t - self.prev_t
        if dt <= 0.0:
            # Non-increasing timestamp; skip
            return

        # Linear velocity (in marker_rotated frame)
        v = (p - self.prev_p) / dt

        # Angular velocity:
        # Relative rotation taking previous frame to current: R_rel = R_prev^T * R_now
        R_rel = self.prev_R.T.dot(R)
        axis, angle = axis_angle_from_R(R_rel)
        omega = (angle / dt) * axis  # rad/s, components are in marker_rotated frame

        # Publish
        tw = TwistStamped()
        tw.header = msg.header  # frame_id should be "marker_rotated"
        tw.twist.linear.x  = float(v[0])
        tw.twist.linear.y  = float(v[1])
        tw.twist.linear.z  = float(v[2])
        tw.twist.angular.x = float(omega[0])
        tw.twist.angular.y = float(omega[1])
        tw.twist.angular.z = float(omega[2])
        self.pub.publish(tw)

        # (Optional) print to terminal occasionally
        rospy.loginfo_throttle(0.2,
            "v[m/s]=[%.4f %.4f %.4f]  ω[rad/s]=[%.4f %.4f %.4f]" %
            (v[0], v[1], v[2], omega[0], omega[1], omega[2])
        )

        # Update state
        self.prev_t, self.prev_p, self.prev_R = t, p, R

def main():
    rospy.init_node("current_velocity_from_transformed_pos")
    VelocityFromPose()
    rospy.spin()

if __name__ == "__main__":
    main()
