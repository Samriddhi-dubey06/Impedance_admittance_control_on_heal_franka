#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from cv_bridge import CvBridge
from tf.transformations import quaternion_from_matrix, euler_from_matrix

class ArucoId6Pose:
    def __init__(self):
        rospy.init_node("aruco_id6_pose", anonymous=True)
        self.bridge = CvBridge()

        # ---------- Params ----------
        self.image_topic       = rospy.get_param("~image_topic",       "/camera/color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.pose_topic        = rospy.get_param("~pose_topic",        "/aruco/id6/pose")
        self.rpy_topic         = rospy.get_param("~rpy_topic",         "/aruco/id6/rpy")
        self.marker_id         = int(rospy.get_param("~marker_id",     6))
        self.marker_length_m   = float(rospy.get_param("~marker_length_m", 0.100))  # 128 mm
        self.window_name       = rospy.get_param("~window_name",       "ArUco ID 6")

        # ---------- Camera intrinsics (from CameraInfo) ----------
        self.K = None
        self.dist = None
        self.out_frame = "camera"
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1)

        # ---------- ArUco (OpenCV 4.2-friendly API) ----------
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.det_params = aruco.DetectorParameters()



        # ---------- ROS I/O ----------
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=10)
        self.rpy_pub  = rospy.Publisher(self.rpy_topic,  Vector3Stamped, queue_size=10)
        rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1)

        rospy.loginfo("ArucoId6Pose: image=%s, camera_info=%s, marker_id=%d, size=%.3f m",
                      self.image_topic, self.camera_info_topic, self.marker_id, self.marker_length_m)

    def _camera_info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.array(msg.D, dtype=np.float64).reshape(-1)
        if msg.header.frame_id:
            self.out_frame = msg.header.frame_id

    def _image_cb(self, msg: Image):
        if self.K is None or self.dist is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge error: %s", e)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.det_params)

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)
            ids = ids.flatten()

            # find our marker id
            idx = np.where(ids == self.marker_id)[0]
            if len(idx) > 0:
                i = int(idx[0])
                s = self.marker_length_m

                # square corners in marker frame (Z=0)
                obj_pts = np.array(
                    [[-s/2,  s/2, 0],
                     [ s/2,  s/2, 0],
                     [ s/2, -s/2, 0],
                     [-s/2, -s/2, 0]], dtype=np.float32
                )
                img_pts = corners[i].reshape(-1, 2)

                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.dist)
                if not ok:
                    # nothing to publish/draw
                    self._show(frame)
                    return

                # draw axes for visual check
                try:
                    cv2.drawFrameAxes(frame, self.K, self.dist, rvec, tvec, s * 0.5)
                except AttributeError:
                    # fallback for older OpenCV contrib
                    aruco.drawAxis(frame, self.K, self.dist, rvec, tvec, s * 0.5)

                # rotation & homogeneous transform (marker in camera frame)
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3,  3] = tvec.flatten()

                # quaternion + euler
                qx, qy, qz, qw = quaternion_from_matrix(T)
                roll, pitch, yaw = euler_from_matrix(T, axes='szyx')

                # publish pose (position + quaternion)
                pose = PoseStamped()
                pose.header.stamp = msg.header.stamp
                pose.header.frame_id = self.out_frame
                pose.pose.position.x = float(tvec[0])
                pose.pose.position.y = float(tvec[1])
                pose.pose.position.z = float(tvec[2])
                pose.pose.orientation.x = float(qx)
                pose.pose.orientation.y = float(qy)
                pose.pose.orientation.z = float(qz)
                pose.pose.orientation.w = float(qw)
                self.pose_pub.publish(pose)

                # publish rpy (rad)
                rpy = Vector3Stamped()
                rpy.header = pose.header
                rpy.vector.x = float(roll)
                rpy.vector.y = float(pitch)
                rpy.vector.z = float(yaw)
                self.rpy_pub.publish(rpy)

        self._show(frame)

    def _show(self, frame):
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

def main():
    node = ArucoId6Pose()
    try:
        rospy.spin()
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
