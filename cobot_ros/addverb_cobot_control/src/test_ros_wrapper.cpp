/**
 * @file Test for ros wrapper
 * @author Gonna Yaswanth (yaswant.gonna@addverb.com)
 * @brief Test client for the joint veloctiy controller action server
 * @version 0.1
 * @date 2024-08-22
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <vector>
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <api_types.h>
#include "control_msgs/FollowJointTrajectoryAction.h"
#include <sstream>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "joint_velocity_trajectory_client");

    actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> ac_("velocity_trajectory_controller/follow_joint_trajectory", true);

    ROS_INFO_STREAM("Waiting for action server to start");

    ac_.waitForServer();

    ROS_INFO_STREAM("Action server started");

    control_msgs::FollowJointTrajectoryGoal goal;
    
    for(int joint_id = 0; joint_id < 6; joint_id++)
    {
        std::stringstream jointName;
        jointName << "Joint" << "_" << joint_id+1;
        goal.trajectory.joint_names.push_back(jointName.str());
    }
    
    // Trajectory has only two points
    int traj_id = 0;

    // Setting joint velocities
    goal.trajectory.points[traj_id].velocities.resize(6);
    goal.trajectory.points[traj_id].velocities[0] = 0.0;
    goal.trajectory.points[traj_id].velocities[1] = 0.0;
    goal.trajectory.points[traj_id].velocities[2] = 0.0;
    goal.trajectory.points[traj_id].velocities[3] = 0.0;
    goal.trajectory.points[traj_id].velocities[4] = 0.0;
    goal.trajectory.points[traj_id].velocities[5] = 0.0;
    goal.trajectory.points[traj_id].velocities[6] = 0.0;
    
    traj_id += 1;
    goal.trajectory.points[traj_id].velocities.resize(6);
    goal.trajectory.points[traj_id].velocities[0] = 0.001;
    goal.trajectory.points[traj_id].velocities[1] = 0.0;
    goal.trajectory.points[traj_id].velocities[2] = 0.0;
    goal.trajectory.points[traj_id].velocities[3] = 0.0;
    goal.trajectory.points[traj_id].velocities[4] = 0.0;
    goal.trajectory.points[traj_id].velocities[5] = 0.0;
    goal.trajectory.points[traj_id].velocities[6] = 0.0;
    goal.trajectory.points[traj_id].time_from_start = ros::Duration(2);

    traj_id += 1;
    goal.trajectory.points[traj_id].velocities.resize(6);
    goal.trajectory.points[traj_id].velocities[0] = 0.0;
    goal.trajectory.points[traj_id].velocities[1] = 0.0;
    goal.trajectory.points[traj_id].velocities[2] = 0.0;
    goal.trajectory.points[traj_id].velocities[3] = 0.0;
    goal.trajectory.points[traj_id].velocities[4] = 0.0;
    goal.trajectory.points[traj_id].velocities[5] = 0.0;
    goal.trajectory.points[traj_id].velocities[6] = 0.0;
    goal.trajectory.points[traj_id].time_from_start = ros::Duration(2);

    // sending goal
    ac_.sendGoal(goal);

    // wait for the expected amount of time
    bool could_complete = ac_.waitForResult(ros::Duration(10)); //goal.controller_config.delta_t + 20));

    if (could_complete)
    {
        ROS_INFO_STREAM("Robot reached at the target velocity");
    }
    else
    {
        ROS_INFO_STREAM("Robot could not reach at the target velocity");
    }

    return 0;
}