#!/usr/bin/env python3
import rospy
import actionlib

from control_msgs.msg import GripperCommandAction, GripperCommandGoal

from addverb_cobot_msgs.msg import (
    GraspAction as HealGraspAction,
    GraspGoal as HealGraspGoal,
    ReleaseAction as HealReleaseAction,
    ReleaseGoal as HealReleaseGoal,
)

class HealGripperSimpleController:
    def __init__(self):
        self.grasp_client = actionlib.SimpleActionClient("/heal/robotA/grasp_action", HealGraspAction)
        self.release_client = actionlib.SimpleActionClient("/heal/robotA/release_action", HealReleaseAction)

        rospy.loginfo("Waiting for HEAL gripper action servers...")
        self.grasp_client.wait_for_server()
        self.release_client.wait_for_server()
        rospy.loginfo("HEAL gripper action servers ready.")

    def close_gripper(self, grasp_force=100):
        goal = HealGraspGoal()
        goal.grasp_force = grasp_force
        rospy.loginfo(f"HEAL: Closing gripper with force {grasp_force} N")
        self.grasp_client.send_goal(goal)
        self.grasp_client.wait_for_result()
        rospy.loginfo("HEAL gripper closed.")

    def open_gripper(self):
        goal = HealReleaseGoal()
        rospy.loginfo("HEAL: Opening gripper.")
        self.release_client.send_goal(goal)
        self.release_client.wait_for_result()
        rospy.loginfo("HEAL gripper opened.")


class FrankaGripperSimpleController:
    def __init__(self):
        self.client = actionlib.SimpleActionClient("/fr3/franka_gripper/gripper_action", GripperCommandAction)

        rospy.loginfo("Waiting for Franka gripper action server...")
        self.client.wait_for_server()
        rospy.loginfo("Franka gripper action server ready.")

    def open_gripper(self, width=0.08, effort=10.0):
        goal = GripperCommandGoal()
        goal.command.position = width
        goal.command.max_effort = effort

        rospy.loginfo(f"Franka: Opening gripper to {width:.2f} m, effort {effort}")
        self.client.send_goal(goal)
        self.client.wait_for_result()
        rospy.loginfo("Franka gripper opened.")

    def close_gripper(self, width=0.02, effort=40.0):
        goal = GripperCommandGoal()
        goal.command.position = width
        goal.command.max_effort = effort

        rospy.loginfo(f"Franka: Closing gripper to {width:.2f} m, effort {effort}")
        self.client.send_goal(goal)
        self.client.wait_for_result()
        rospy.loginfo("Franka gripper closed.")

if __name__ == "__main__":
    try:
        rospy.init_node("gripper_controller", anonymous=True)

        fr3_gripper = FrankaGripperSimpleController()
        fr3_gripper.open_gripper(width=0.08, effort=10)

        # heal_gripper = HealGripperSimpleController()
        # heal_gripper.close_gripper(grasp_force=100)

    except rospy.ROSInterruptException:
        pass