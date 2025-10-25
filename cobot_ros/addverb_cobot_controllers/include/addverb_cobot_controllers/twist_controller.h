#ifndef ADDVERB_TWIST_CONTROLLER_H
#define ADDVERB_TWIST_CONTROLLER_H

#include <geometry_msgs/TwistStamped.h>
#include <realtime_tools/realtime_buffer.h>
#include <controller_interface/controller.h>
#include "twist_command_handle.h"
#include <pluginlib/class_list_macros.hpp>

namespace addverb
{
    class TwistController : public controller_interface::Controller<TwistCommandInterface>
    {
    public:
        /// @brief ctor
        TwistController() = default;

        /// @brief dtor
        ~TwistController() {};

        /// @brief implementation for the controller_manager to initialise this controller
        bool init(TwistCommandInterface *, ros::NodeHandle &) override;

        /// @brief implementation for the controller_manager to start this controller
        void starting(const ros::Time &) override;

        /// @brief update the latest user input to the hardware
        void update(const ros::Time &, const ros::Duration &);

        /// @brief implementation for the controller_manager to initialise this controller
        void stopping(const ros::Time &) {};

    private:
        /// @brief subscriber to handle user input
        ros::Subscriber twist_sub_;

        /// @brief handle for twist controller
        TwistCommandHandle handler_;

        /// @brief buffer to handle user input
        realtime_tools::RealtimeBuffer<geometry_msgs::Twist> command_buffer_;

        /// @brief callback handling data given by user
        void twistCallback(const geometry_msgs::TwistConstPtr &);

        /// @brief test validity of input twist command
        bool isValid_(const geometry_msgs::TwistConstPtr &);

        /// @brief test validity of input command
        bool isValid_(const geometry_msgs::Vector3 &);

        /// @brief test validity of input command
        bool inRange_(const double &);
    };
} // namespace addverb

#endif