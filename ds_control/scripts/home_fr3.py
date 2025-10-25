#!/usr/bin/env python3
"""
home_fr3.py

Move the FR3 arm from its current pose to a “home” 7 joint configuration
using a quintic joint space trajectory and velocity control.
"""

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from ds_control.trajectory_planner import TrajectoryPlanner

# “Home” joint angles for FR3 (7‐DOF)
HOME_FR3_JOINTS    = np.array([
    0.0015584473835553482,
   -0.7841060606330866,
   -0.0007111409103181678,
   -2.357667109743753,
   -0.0006936037327182386,
    1.5735468349554234,
    0.7782996616799904
], dtype=float)
TRAJECTORY_TIME    = 5.0   # seconds
DT                 = 0.001 # 1 kHz

class HomeFR3Commander:
    def __init__(self):
        rospy.init_node("home_fr3", anonymous=True)

        # Publisher to FR3’s joint‐velocity controller
        self.pub = rospy.Publisher(
            "/fr3/joint_velocity_controller/joint_velocity_command",
            Float64MultiArray, queue_size=1
        )
        # Subscriber to FR3 joint states
        self.sub = rospy.Subscriber(
            "/fr3/joint_states",
            JointState,
            self.joint_state_callback
        )

        self.current_joints    = None
        self.planner           = TrajectoryPlanner()
        self.traj_generated    = False
        self.velocity_traj     = None
        self.dt                = DT
        self.trajectory_index  = 0

    def joint_state_callback(self, msg: JointState):
        # read first 7 joints
        positions = np.array(msg.position)
        if positions.size < 7:
            rospy.logwarn("home_fr3: received fewer than 7 joints")
            return
        self.current_joints = positions[:7]

        # generate trajectory once
        if not self.traj_generated:
            _, self.velocity_traj, _ = self.planner.quintic_joint_trajectory(
                self.current_joints,
                HOME_FR3_JOINTS,
                TRAJECTORY_TIME,
                self.dt
            )
            self.traj_generated   = True
            self.trajectory_index = 0
            rospy.loginfo(f"home_fr3: trajectory from {self.current_joints} to {HOME_FR3_JOINTS}")

    def run(self):
        rate = rospy.Rate(1.0 / self.dt)
        while not rospy.is_shutdown():
            if self.traj_generated and self.trajectory_index < len(self.velocity_traj):
                vel_cmd = self.velocity_traj[self.trajectory_index]
                msg = Float64MultiArray(data=vel_cmd.tolist())
                self.pub.publish(msg)
                self.trajectory_index += 1
            rate.sleep()

if __name__ == "__main__":
    node = HomeFR3Commander()
    node.run()
