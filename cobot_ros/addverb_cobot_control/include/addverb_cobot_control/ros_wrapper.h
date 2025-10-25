/**
 * @file ros_wrapper.cpp
 * @author Gonna Yaswanth (yaswanth.gonna@addverb.com)
 * @brief Cobot ROS wrapper
 * @version 0.1
 * @date 2024-09-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef ADDVERB_ROS_WRAPPER_H_
#define ADDVERB_ROS_WRAPPER_H_

// ROS header files
#include <ros/ros.h>
#include <ros/callback_queue.h>

#include <actionlib/server/simple_action_server.h>
#include <addverb_cobot_msgs/ErrorRecoveryAction.h>
#include <addverb_cobot_msgs/GraspAction.h>
#include <addverb_cobot_msgs/ReleaseAction.h>
#include <controller_manager/controller_manager.h>
#include "std_srvs/SetBool.h"

// Addverb cobot hardware interface
#include "addverb_cobot_hw.h"

// Data handler
#include <data_handler.h>
#include <api_types.h>
#include "ui_data_types.h"

// util header
#include <time.h>
#include <chrono>
#include <memory>
#include <functional>
#include <atomic>
#include <mutex>
#include <limits>

namespace addverb
{
    namespace safety_limits
    {
        /// @brief lower limit of mass at end-effector
        const double mass_ll = 0.1;

        /// @brief upper limit of mass at end-effector
        const double mass_ul = 4.9;
    }
};

namespace addverb
{
    class RosWrapper
    {
    public:
        /// @brief sets up action_servers, loads configuration from parameter server and configures the robot accordingly
        /// @param nh
        /// @param handler
        RosWrapper(ros::NodeHandle &nh, std::shared_ptr<DataHandler> &handler);

        /// @brief dtor - default exit routine
        ~RosWrapper()
        {
            shutdown();
        };

        /// @brief shutdown connection with robot
        bool shutdown();

    private:
        /// @brief ROS control controller frequency
        const int loop_hz_ = 30;

        /// @brief Max number of connection attempts with server
        const int n_connect_attempts_ = 5;

        /// @brief flag to indicate if system should shutdown
        bool should_shutdown_ = false;

        /// @brief status of payload being attached to end-effector of the robot
        bool has_payload_ = false;

        /// @brief flag to be updated when the robot goes to error
        std::atomic<bool> has_error_{false};

        /// @brief mutex lock for data_handler
        std::mutex mut_;

        /// @brief Datahandler for sending command and reading states
        std::shared_ptr<DataHandler> handler_;

        /// @brief ROS control manager
        std::shared_ptr<controller_manager::ControllerManager> cm_;

        /// @brief funrtion to run on loop
        std::function<void()> run_;

        /// @brief ROS node handler
        ros::NodeHandle nh_;

        /// @brief Addverb hardware interface
        std::shared_ptr<AddverbCobotHw> hw_;

        /// @brief To keep track of time for controller
        struct timespec last_time_;

        struct timespec current_time_;

        ros::Duration elapsed_time_;

        /// @brief Robot feedback from API
        RobotFeedback robot_fdbk_;

        /// @brief External control
        ControlInterrupt c_interrupt_;

        /// @brief launch controller config
        ControllerConfig cconf_;

        /// @brief For running control loop in seperate thread
        std::thread *ros_control_thread_;

        /// @brief Service to stop the robot and disconnect from server
        ros::ServiceServer shutdown_robot_srv_;

        /// @brief Action server for error recovery
        std::unique_ptr<actionlib::ActionServer<addverb_cobot_msgs::ErrorRecoveryAction>> err_recovery_as_;

        /// @brief Action server for grasp action
        std::unique_ptr<actionlib::ActionServer<addverb_cobot_msgs::GraspAction>> grasp_as_;

        /// @brief Action server for release action
        std::unique_ptr<actionlib::ActionServer<addverb_cobot_msgs::ReleaseAction>> release_as_;

        /// @brief goal handle for updating the goal status
        actionlib::ServerGoalHandle<addverb_cobot_msgs::ErrorRecoveryAction> goal_handle_;

        /// @brief Payload configuration
        PayloadConfig pconf_;

        /// @brief state of the robot
        RobotState state_;

        /// @brief FLag to check if payload param are loaded
        bool payload_status_;

        /// @brief api given by user
        API api_;

        /// @brief current time point
        ros::Time now_;

        /// @brief record of the previous gripper event executed
        std::atomic<int> gripper_status_{static_cast<int>(Event::eNone)};

        /// @brief controller map
        std::map<API, int> controller_safety_mode_map_; 


        /// @brief trajectory control
        void trajControl_();

        void controlLoop_();

        void rosControl_();

        bool goToMotionState_();

        bool shutdownRobotCB_(std_srvs::SetBool::Request &req,
                              std_srvs::SetBool::Response &res);

        bool resetController_();

        void checkForError_();

        void errRecoveryGoalCB_(
            actionlib::ServerGoalHandle<
                addverb_cobot_msgs::ErrorRecoveryAction>);

        void errRecoveryCancelCB_(
            actionlib::ServerGoalHandle<
                addverb_cobot_msgs::ErrorRecoveryAction>);

        void graspGoalCB_(
            actionlib::ServerGoalHandle<
                addverb_cobot_msgs::GraspAction>);

        void graspCancelCB_(
            actionlib::ServerGoalHandle<
                addverb_cobot_msgs::GraspAction>);

        void releaseGoalCB_(
            actionlib::ServerGoalHandle<
                addverb_cobot_msgs::ReleaseAction>);

        void releaseCancelCB_(
            actionlib::ServerGoalHandle<
                addverb_cobot_msgs::ReleaseAction>);

        /// @brief update API based on recd string updated in param server
        bool updateAPI_(const std::string &);

        /// @brief flag to indicate if
        bool shouldResetController_();

        /// @brief update the payload configuration to the robot, if payload is attached to the robot end-effector
        /// @return
        bool updatePayload_();

        /// @brief check for validity of payload
        /// @return
        bool isValidPayload_(const PayloadConfig &);

        /// @brief get the payload configuration from the parameter file
        /// @param
        void getPayload_(PayloadConfig &);

        /// @brief get safety mode from controller api type
        void getSafetyMode_();
    };
}

#endif