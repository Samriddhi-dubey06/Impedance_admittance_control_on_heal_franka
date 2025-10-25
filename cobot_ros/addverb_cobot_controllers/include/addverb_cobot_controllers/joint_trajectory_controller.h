/**
 * @file joint_trajectory_controller.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-09-30
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef ADDVERB_JOINT_TRAJECTORY_CONTROLLER_H_
#define ADDVERB_JOINT_TRAJECTORY_CONTROLLER_H_

#include <controller_interface/controller.h>
#include <pluginlib/class_list_macros.hpp>
#include "joint_trajectory_command_handle.h"
#include <trajectory_msgs/JointTrajectory.h>

#include <queue>
#include <mutex>

namespace addverb
{
    class JointTrajectoryController : public controller_interface::Controller<JointTrajectoryCommandInterface>
    {
    public:

        /// @brief ctor
        JointTrajectoryController() = default;

        /// @brief dtor
        ~JointTrajectoryController() {};

        /// @brief provide method for controller_manager to initialise controller
        bool init(JointTrajectoryCommandInterface *, ros::NodeHandle &) override;

        /// @brief provide method for controller_manager to start controller
        void starting(const ros::Time &) {};

        /// @brief update control input based on time
        void update(const ros::Time &, const ros::Duration &) override;

        /// @brief provide method for controller_manager to stop controller
        void stopping(const ros::Time &time) {};

    private:

        /// @brief handler to interact with HWInterface
        JointTrajectoryCommandHandle handler_;

        /// @brief queue to maintain the trajectories updated at run-time
        std::queue<trajectory_msgs::JointTrajectory> traj_queue_;

        /// @brief subscriber that updates the trajectory provided by client
        ros::Subscriber traj_sub_;

        /// @brief mutex 
        std::mutex mut_;

        /// @brief callback to handle updates to desired trajectory
        void trajCB_(const trajectory_msgs::JointTrajectoryConstPtr &msg);

        /// @brief update null trajectory to the command handle
        void updateNullTraj_();

        /// @brief check for validity of user-provided input
        bool isValid_(const trajectory_msgs::JointTrajectoryConstPtr &);

        /// @brief check validity of each point 
        bool isValid_(const trajectory_msgs::JointTrajectoryPoint&);
    };
} // namespace addverb

#endif