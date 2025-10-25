#include "twist_controller.h"

namespace addverb
{
    /**
     * @brief implementation for the controller_manager to initialise this controller
     *
     * @param hw
     * @param nh
     * @return true
     * @return false
     */
    bool TwistController::init(TwistCommandInterface *hw, ros::NodeHandle &nh)
    {
        std::string frame_id;

        if (!nh.getParam("frame_id", frame_id))
        {
            ROS_ERROR("No frame_id was given");
            return false;
        }

        handler_ = hw->getHandle(frame_id);
        twist_sub_ = nh.subscribe<geometry_msgs::Twist>("command", 1, &TwistController::twistCallback, this);

        std::vector<std::string> joint_names;

        if (!nh.getParam("joints", joint_names))
        {
            ROS_ERROR("No joint name were provided");
            return false;
        }

        return true;
    }

    /**
     * @brief implementation for the controller_manager to start this controller
     *
     * @param time
     */
    void TwistController::starting(const ros::Time &time)
    {
        geometry_msgs::Twist twist;
        twist.linear.x = 0;
        twist.linear.y = 0;
        twist.linear.z = 0;
        twist.angular.x = 0;
        twist.angular.y = 0;
        twist.angular.z = 0;
        command_buffer_.writeFromNonRT(twist);
    }

    /**
     * @brief callback to be executed once message is updated to subscriber
     *
     * @param msg
     */
    void TwistController::twistCallback(const geometry_msgs::TwistConstPtr &msg)
    {
        if (isValid_(msg))
        {
            geometry_msgs::Twist twist;
            twist = *msg;
            command_buffer_.writeFromNonRT(twist);
        }
    }

    /**
     * @brief update the latest user input to the hardware
     *
     */
    void TwistController::update(const ros::Time &, const ros::Duration &)
    {
        handler_.setCommand(*command_buffer_.readFromRT());
    };

    /**
     * @brief test validity of input twist command
     *
     * @param msg
     * @return true
     * @return false
     */
    bool TwistController::isValid_(const geometry_msgs::TwistConstPtr &msg)
    {
        geometry_msgs::Twist twist = *msg;

        if (!isValid_(twist.linear))
        {
            ROS_WARN("Invalid twist command. Please ensure the values belong to the set {-1,0,1}");
            return false;
        }

        if (!isValid_(twist.angular))
        {
            ROS_WARN("Invalid twist command. Please ensure the values belong to the set {-1,0,1}");
            return false;
        }

        return true;
    }

    /**
     * @brief test validity of input command
     *
     * @param msg
     * @return true
     * @return false
     */
    bool TwistController::isValid_(const geometry_msgs::Vector3 &vec)
    {
        bool out = inRange_(vec.x);

        out = out && inRange_(vec.y);

        out = out && inRange_(vec.z);

        return out;
    }

    /**
     * @brief test validity of input command
     *
     * @param val
     * @return true
     * @return false
     */
    bool TwistController::inRange_(const double &val)
    {
        return ((val <= 1) && (val >= -1));
    }

}

PLUGINLIB_EXPORT_CLASS(addverb::TwistController, controller_interface::ControllerBase)