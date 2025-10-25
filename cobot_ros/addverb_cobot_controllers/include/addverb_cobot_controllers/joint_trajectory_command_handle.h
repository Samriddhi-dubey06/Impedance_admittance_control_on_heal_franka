
#ifndef ADDVERB_JOINT_TRAJECTORY_COMMAND_HANDLE_H_
#define ADDVERB_JOINT_TRAJECTORY_COMMAND_HANDLE_H_
#include <ros/ros.h>
#include "trajectory_msgs/JointTrajectory.h"
#include <hardware_interface/internal/hardware_resource_manager.h>

namespace addverb
{

	class JointTrajectoryCommandHandle
	{
	public:
		JointTrajectoryCommandHandle() = default;

		JointTrajectoryCommandHandle(std::string &name, trajectory_msgs::JointTrajectory *trajectory, bool *cancel_goal, bool *offload_traj) : name_(name), cmd_(trajectory), cancel_goal_(cancel_goal), offload_traj_(offload_traj)
		{
			if (!cmd_)
			{
				ROS_ERROR("Command pointer was null can not create handle");
			}
			if (!cancel_goal_)
			{
				ROS_ERROR("Cancel goal pointer was null can not create handler");
			}
			if (!offload_traj_)
			{
				ROS_ERROR("Offload trajectory pointer was null. Can not create handler");
			}
		}

		~JointTrajectoryCommandHandle() = default;

		std::string getName() const
		{
			return name_;
		}

		void setCommand(const trajectory_msgs::JointTrajectory &joint_trajectory)
		{
			assert(cmd_);
			*cmd_ = joint_trajectory;
		}

		trajectory_msgs::JointTrajectory getCommand() const
		{
			assert(cmd_);
			return *cmd_;
		}

		const trajectory_msgs::JointTrajectory *getCommandPtr() const
		{
			assert(cmd_);
			return cmd_;
		}

		void setGoalCanceled(const bool cg)
		{
			assert(cg);
			*cancel_goal_ = cg;
		}

		bool shouldOffloadTraj()
		{
			assert(offload_traj_);
			return (*offload_traj_);
		}

	private:
		trajectory_msgs::JointTrajectory *cmd_ = {nullptr};
		bool *cancel_goal_ = {nullptr};
		bool *offload_traj_ = {nullptr};
		std::string name_;
	};

	class JointTrajectoryCommandInterface
		: public hardware_interface::HardwareResourceManager<JointTrajectoryCommandHandle, hardware_interface::ClaimResources>
	{
	};
} // namespace addverb

#endif