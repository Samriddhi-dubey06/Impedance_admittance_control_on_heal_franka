#!/usr/bin/env python3

import rospy
import moveit_commander
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped

class PlanAndVelocityPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('plan_and_velocity_publisher', anonymous=True)

        # Initialize MoveIt
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("heal_arm")  # Replace with your group name

        # Configure MoveIt planning parameters
        self.group.set_planner_id("RRTConnectkConfigDefault")
        self.group.set_planning_time(10.0)
        self.group.set_num_planning_attempts(10)
        self.group.allow_replanning(True)
        self.group.set_max_velocity_scaling_factor(0.1)
        self.group.set_max_acceleration_scaling_factor(0.1)

        rospy.loginfo("Waiting for MoveIt to be ready...")
        rospy.wait_for_service('/get_planning_scene')
        rospy.loginfo("MoveIt is ready.")

        # Publisher for velocity commands (assumes a JointGroupVelocityController on this topic)
        self.velocity_pub = rospy.Publisher('/velocity_controller/command', Float64MultiArray, queue_size=10)

    def plan_and_execute(self):
        """
        1) Plans a trajectory to move the end effector +0.2m in X,
        2) Prints time, positions, velocities,
        3) Publishes velocity commands to /velocity_controller/command.
        """
        # Make sure we start from the current robot state
        self.group.set_start_state_to_current_state()

        # Get the current end-effector pose
        ee_pose = self.group.get_current_pose().pose
        rospy.loginfo(f"Current EE Pose: x={ee_pose.position.x}, y={ee_pose.position.y}, z={ee_pose.position.z}")

        # Define a new pose: move +0.2m in X
        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_link"
        target_pose.pose = ee_pose
        target_pose.pose.position.x += 0.2

        # Set the target pose
        self.group.set_pose_target(target_pose.pose)

        # Plan the trajectory
        rospy.loginfo("Planning the trajectory...")
        plan_output = self.group.plan()

        # Parse the planning result
        success = False
        traj = None
        if isinstance(plan_output, tuple):
            # Some MoveIt! versions: (success_flag, RobotTrajectory, planning_time, error_code)
            success, traj, planning_time, error_code = plan_output
            rospy.loginfo(f"Planning time: {planning_time}, Error code: {error_code}")
        else:
            # Other versions may return just a RobotTrajectory or MoveItErrorCode
            success = True
            traj = plan_output

        if not success or not traj:
            rospy.logerr("Planning failed or returned an empty trajectory.")
            return

        rospy.loginfo("Trajectory planned successfully!")

        joint_traj = traj.joint_trajectory
        self.print_trajectory_info(joint_traj)
        self.publish_velocity_commands(joint_traj)

    def print_trajectory_info(self, joint_traj):
        """
        Prints each trajectory point's time_from_start, positions, and velocities.
        """
        rospy.loginfo("=== Trajectory Points ===")
        for i, point in enumerate(joint_traj.points):
            t = point.time_from_start.to_sec()
            positions = point.positions
            velocities = point.velocities
            rospy.loginfo(
                f"Point {i}: time={t:.3f} sec, "
                f"positions={positions}, velocities={velocities}"
            )

    def publish_velocity_commands(self, joint_traj):
        """
        Converts each trajectory point's velocities into a message for the
        velocity controller, publishing them at ~100Hz.
        """
        rate_hz = 100
        rate = rospy.Rate(rate_hz)
        rospy.loginfo("Publishing velocity commands to /velocity_controller/command...")

        for i, point in enumerate(joint_traj.points):
            # Create velocity message
            velocity_msg = Float64MultiArray()
            velocity_msg.data = list(point.velocities)  # Convert tuple to list

            # Log for debugging
            rospy.loginfo(f"Publishing velocities for point {i}: {velocity_msg.data}")

            # Publish velocity
            self.velocity_pub.publish(velocity_msg)

            rate.sleep()

        # Stop the robot by sending zero velocities
        rospy.loginfo("Stopping the robot with zero velocity...")
        zero_msg = Float64MultiArray(data=[0.0]*len(joint_traj.joint_names))
        self.velocity_pub.publish(zero_msg)

if __name__ == '__main__':
    try:
        node = PlanAndVelocityPublisher()
        node.plan_and_execute()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()
