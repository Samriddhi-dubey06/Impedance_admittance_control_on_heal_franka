#!/usr/bin/env python3
"""
home_heal.py

Move the HEAL arm from its current pose to a “home” 6 joint configuration
using a quintic joint space trajectory and velocity control.
"""

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from ds_control.trajectory_planner import TrajectoryPlanner

# “Home” joint angles for HEAL (6‐DOF)
HOME_HEAL_JOINTS    = np.zeros(6, dtype=float)
TRAJECTORY_TIME     = 8.0   # seconds
DT                  = 0.01 # 1 kHz

class HomeHealCommander:
    def __init__(self):
        rospy.init_node("home_heal", anonymous=True)

        # Publisher to HEAL’s joint‐velocity controller
        self.pub = rospy.Publisher(
            "/heal/velocity_controller/command",
            Float64MultiArray, queue_size=1
        )
        # Subscriber to HEAL joint states
        self.sub = rospy.Subscriber(
            "/heal/joint_states",
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
        positions = np.array(msg.position)
        if positions.size < 6:
            rospy.logwarn("home_heal: received fewer than 6 joints")
            return
        self.current_joints = positions[:6]

        if not self.traj_generated:
            _, self.velocity_traj, _ = self.planner.quintic_joint_trajectory(
                self.current_joints,
                HOME_HEAL_JOINTS,
                TRAJECTORY_TIME,
                self.dt
            )
            self.traj_generated   = True
            self.trajectory_index = 0
            rospy.loginfo(f"home_heal: trajectory from {self.current_joints} to {HOME_HEAL_JOINTS}")

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
    node = HomeHealCommander()
    node.run()
