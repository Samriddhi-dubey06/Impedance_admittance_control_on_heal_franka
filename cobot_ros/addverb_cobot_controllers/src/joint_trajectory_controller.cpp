#include "joint_trajectory_controller.h"

namespace addverb
{

    /**
     * @brief initialise the controller - basically acquire the resource handle
     *
     * @param hw
     * @param nh
     * @return true
     * @return false
     */
    bool JointTrajectoryController::init(JointTrajectoryCommandInterface *hw, ros::NodeHandle &nh)
    {
        std::string controller_type;
        ros::NodeHandle nh_private = ros::NodeHandle(nh, "joint_trajectory_controller"); // Scoped Namespace
        if (!nh_private.getParam("ros_control_mode", controller_type))
        {
            ROS_ERROR("No controller type was provided");
            return false;
        }

        handler_ = hw->getHandle(controller_type);

        traj_sub_ = nh.subscribe<trajectory_msgs::JointTrajectory>("command", 1, &JointTrajectoryController::trajCB_, this);

        return true;
    }

    /**
     * @brief update the next trajectory , if available,
     * if not available, the calling method is notified by updating the desired time = -1
     *
     */
    void JointTrajectoryController::update(const ros::Time &, const ros::Duration &)
    {
        if (handler_.shouldOffloadTraj())
        {
            bool has_new_traj = false;
            trajectory_msgs::JointTrajectory next_traj;

            {
                std::lock_guard<std::mutex> guard(mut_);
                if (!traj_queue_.empty())
                {
                    has_new_traj = true;
                    next_traj = traj_queue_.front();
                    traj_queue_.pop();
                }
            }

            if (has_new_traj)
            {
                handler_.setCommand(next_traj);
            }
            else
            {
                updateNullTraj_();
            }
        }
        else
        {
            updateNullTraj_();
        }
    };

    /**
     * @brief handle updates to desired trajectory
     *
     * @param msg
     */
    void JointTrajectoryController::trajCB_(const trajectory_msgs::JointTrajectoryConstPtr &msg)
    {

        if (!isValid_(msg))
        {
            return;
        }

        ROS_INFO("Queuing up new trajectory");
        {
            std::lock_guard<std::mutex> guard(mut_);

            traj_queue_.push(*msg);
        }
    }

    /**
     * @brief update null trajectory to the command handle
     *
     */
    void JointTrajectoryController::updateNullTraj_()
    {
        trajectory_msgs::JointTrajectory traj;
        trajectory_msgs::JointTrajectoryPoint point;
        point.time_from_start = ros::Duration(-1);
        traj.points.push_back(point);
        handler_.setCommand(traj);
    }

    /**
     * @brief validity checks for user provided input
     *
     * @param traj
     * @return true
     * @return false
     */
    bool JointTrajectoryController::isValid_(const trajectory_msgs::JointTrajectoryConstPtr &msg)
    {
        trajectory_msgs::JointTrajectory traj = *msg;

        if (traj.points.size() > 1)
        {
            if (traj.points.size() < 3)
            {
                ROS_WARN("Please specify a minimum of three points if you wish to run a Multi-Point control");
                return false;
            }
        }

        for (uint i = 0; i < traj.points.size(); i++)
        {
            if (!isValid_(traj.points[i]))
            {
                return false;
            }

            // additional checks for multi-point config
            if (traj.points.size() > 1)
            {
                double t_prev = 0;
                if (i != 0)
                {
                    t_prev = traj.points[i - 1].time_from_start.toSec();
                }

                if (traj.points[i].time_from_start.toSec() <= t_prev)
                {
                    ROS_WARN("Time for attaining target position in Multi-Point configuration must be incremental, i.e., Please ensure that for (i+1)th target position time specified must be greater than time specified for ith target position");
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * @brief validity checks for user provided input
     *
     * @param traj
     * @return true
     * @return false
     */
    bool JointTrajectoryController::isValid_(const trajectory_msgs::JointTrajectoryPoint &point)
    {
        if (point.positions.size() != 6)
        {
            ROS_WARN("Target position must be specified for each joint. Incomplete target specified");
            return false;
        }

        for (uint i = 0; i < 6; i++)
        {
            double check_lim = 3.14;
            if (std::isnan(point.positions[i]))
            {
                ROS_WARN("Invalid target position specified");
                return false;
            }

            if (std::abs(point.positions[i]) > check_lim)
            {
                ROS_WARN("Invalid target position specified");
                return false;
            }
        }

        // additional checks for joint 2 and joint 3
        if (std::abs(point.positions[1]) > 1.57)
        {
            ROS_WARN("Invalid target position specified");
            return false;
        }

        if (point.positions[2] < -0.69777 || point.positions[2] > 1.57)
        {
            ROS_WARN("Invalid target position specified");
            return false;
        }

        double t = point.time_from_start.toSec();
        if ((t <= 5) || std::isnan(t))
        {
            ROS_WARN("Invalid time specified");
            return false;
        }

        return true;
    }

}
PLUGINLIB_EXPORT_CLASS(addverb::JointTrajectoryController, controller_interface::ControllerBase)
