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
        self.group = moveit_commander.MoveGroupCommander("heal_arm")  # Replace with your MoveIt group

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

        # Publisher for velocity commands
        # (Assumes a "JointGroupVelocityController" listening at this topic)
        self.velocity_pub = rospy.Publisher('/velocity_controller/command', Float64MultiArray, queue_size=10)

    def plan_and_execute(self):
        """
        Plans a trajectory to move the end effector +0.2m in X,
        prints the trajectory, and then streams velocity commands
        matching each waypoint's time_from_start.
        """
        # Ensure we start from the robot's current state
        self.group.set_start_state_to_current_state()

        # Get current EE pose
        ee_pose = self.group.get_current_pose().pose
        rospy.loginfo(f"Current EE Pose: x={ee_pose.position.x}, "
                      f"y={ee_pose.position.y}, z={ee_pose.position.z}")

        # Define the target pose: move +0.2m in X
        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_link"
        target_pose.pose = ee_pose
        target_pose.pose.position.x += 0.25

        # Set the target pose
        self.group.set_pose_target(target_pose.pose)

        # Plan the trajectory
        rospy.loginfo("Planning the trajectory...")
        plan_output = self.group.plan()

        # Parse plan result
        success = False
        traj = None
        if isinstance(plan_output, tuple):
            # Some MoveIt! versions return (success_flag, RobotTrajectory, planning_time, error_code)
            success, traj, planning_time, error_code = plan_output
            rospy.loginfo(f"Planning time: {planning_time}, Error code: {error_code}")
        else:
            success = True
            traj = plan_output

        if not success or not traj:
            rospy.logerr("Planning failed or returned an empty trajectory.")
            return

        rospy.loginfo("Trajectory planned successfully!")

        joint_traj = traj.joint_trajectory
        self.print_trajectory_info(joint_traj)
        self.time_based_velocity_execution(joint_traj)

    def print_trajectory_info(self, joint_traj):
        """
        Prints each waypoint's time_from_start, positions, and velocities.
        """
        rospy.loginfo("=== Trajectory Points ===")
        for i, point in enumerate(joint_traj.points):
            t = point.time_from_start.to_sec()
            positions = point.positions
            velocities = point.velocities
            rospy.loginfo(
                f"Point {i}: time={t:.3f}s, positions={positions}, velocities={velocities}"
            )

    def time_based_velocity_execution(self, joint_traj):
        """
        Streams each waypoint's velocity for the correct segment of time
        based on 'time_from_start', ensuring the robot receives continuous
        velocity commands over the duration of the trajectory.
        """
        rate_hz = 100
        rate = rospy.Rate(rate_hz)
        rospy.loginfo("Executing velocity trajectory...")

        points = joint_traj.points
        num_points = len(points)
        n_joints = len(joint_traj.joint_names)

        # If there's only 1 point or none, there's no real motion
        if num_points < 2:
            rospy.logwarn("Trajectory has < 2 points; no motion to execute.")
            return

        # Start time reference
        start_time = rospy.Time.now()

        # Iterate from point i to point i+1
        for i in range(num_points - 1):
            current_pt = points[i]
            next_pt = points[i+1]

            current_vel = current_pt.velocities
            # The duration for this segment:
            seg_duration = (next_pt.time_from_start - current_pt.time_from_start).to_sec()

            # We'll continuously publish the 'current_vel' until we reach the time for the next waypoint
            segment_end_time = rospy.Time.now() + rospy.Duration(seg_duration)

            rospy.loginfo(f"Segment {i} -> {i+1} for {seg_duration:.3f}s, velocities={current_vel}")

            while (rospy.Time.now() < segment_end_time) and not rospy.is_shutdown():
                velocity_msg = Float64MultiArray()
                velocity_msg.data = list(current_vel)
                self.velocity_pub.publish(velocity_msg)

                rate.sleep()

        # Handle the last point (usually velocities=0 or minimal if the trajectory ends stationary)
        last_vel = points[-1].velocities
        rospy.loginfo(f"Last point velocities={last_vel}, publishing briefly...")

        # Publish last velocities for 100ms
        end_buffer_time = rospy.Time.now() + rospy.Duration(0.1)
        while rospy.Time.now() < end_buffer_time and not rospy.is_shutdown():
            velocity_msg = Float64MultiArray()
            velocity_msg.data = list(last_vel)
            self.velocity_pub.publish(velocity_msg)
            rate.sleep()

        # Finally, stop the robot
        rospy.loginfo("Stopping the robot with zero velocity...")
        zero_msg = Float64MultiArray(data=[0.0]*n_joints)
        self.velocity_pub.publish(zero_msg)

if __name__ == '__main__':
    try:
        node = PlanAndVelocityPublisher()
        node.plan_and_execute()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()
