#!/usr/bin/env python3
import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation as R

pub = None

def callback(msg):
    global pub
    # 1) Reconstruct 4×4 in COLUMN-MAJOR (Fortran) order:
    M = np.array(msg.O_T_EE).reshape((4, 4), order='F')
    
    # 2) Extract translation from the last COLUMN:
    position = M[:3, 3]  # [x, y, z]

    # 3) Extract the true 3×3 rotation block:
    R_mat = M[:3, :3]

    # 4) Convert to quaternion [x, y, z, w]:
    quat = R.from_matrix(R_mat).as_quat()

    # 5) Publish
    pose_msg = Pose()
    pose_msg.position.x = position[0]
    pose_msg.position.y = position[1]
    pose_msg.position.z = position[2]
    pose_msg.orientation.x = quat[0]
    pose_msg.orientation.y = quat[1]
    pose_msg.orientation.z = quat[2]
    pose_msg.orientation.w = quat[3]
    pub.publish(pose_msg)

def listener():
    global pub
    rospy.init_node('ee_pose', anonymous=True)
    pub = rospy.Publisher("/fr3/ee_pose", Pose, queue_size=10)
    rospy.Subscriber("/fr3/franka_state_controller/franka_states",
                     FrankaState, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()