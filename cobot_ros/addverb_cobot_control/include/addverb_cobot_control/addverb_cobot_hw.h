/**
 * @file addverb_cobot_hw.h
 * @author Gonna Yaswanth (yaswanth.gonna@addverb.com)
 * @brief Cobot hardware interface for ROS control
 * @version 0.1
 * @date 2024-09-23
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef ADDVERB_COBOT_HW_H
#define ADDVERB_COBOT_HW_H

// ROS
#include <ros/ros.h>

// Data hadnler
#include <data_handler.h>
#include <api_types.h>

// ROS control
#include <hardware_interface/robot_hw.h>
#include <hardware_interface/joint_state_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include "addverb_cobot_controllers/twist_command_handle.h"
#include "addverb_cobot_controllers/joint_trajectory_command_handle.h"
#include <joint_limits_interface/joint_limits_interface.h>
#include <joint_limits_interface/joint_limits.h>
#include <joint_limits_interface/joint_limits_rosparam.h>
#include <controller_manager/controller_manager.h>
#include <sstream>
#include "geometry_msgs/Twist.h"
#include <std_msgs/Float64MultiArray.h>
#include "addverb_cobot_control/friction_modelling.h"



#include <urdf/model.h>
#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/jntarray.hpp>


namespace addverb
{
class AddverbCobotHw : public hardware_interface::RobotHW
{
public:
    /// @brief Construter for ROS node handler, registering interface and datahandler
    /// @param handler
    /// @param nh
    AddverbCobotHw(ros::NodeHandle &nh);


    /// @brief dtor
    ~AddverbCobotHw() {};

    void setControlMode();

    void read(const ros::Time &, const ros::Duration &) override;

    void write(const ros::Time &, const ros::Duration &) override;

    void setter(RobotFeedback &);

    void getter(ControllerConfig &, ControlInterrupt &);

    void getExternalVelocityConfig(ControllerConfig &, ControlInterrupt &);

    void getExternalEffortConfig(ControllerConfig &, ControlInterrupt &);

    void getTeleopConfig(ControlInterrupt &);

    void computeGravityTorques(const std::vector<double> &joint_positions, std::vector<double> &gravity_torques);
    
    void computeMassMatrix(const std::vector<double> &joint_positions, std::vector<std::vector<double>> &mass_matrix);

    void computeCoriolis(const std::vector<double> &joint_positions, const std::vector<double> &joint_velocities, std::vector<double> &coriolis_torques);

    void getLinearVelocityConfig(ControllerConfig &);

    void getControllerConfig(ControllerConfig &);

    void set(const EventStatus &);

    void waitForFirstValidFeedback();

    void performAddCompute();

    /**
     * @brief Check if the hardware interface is in effort control mode.
     * @return true if in effort control mode, false otherwise.
     */
    bool isEffortControlMode() const
    {
        return effort_control_mode_;
    }

    const std::vector<double>& getEffortCommands() const;
    
private:
    /// @brief ROS nodehandler
    ros::NodeHandle nh_;

    /// @brief Joint interface
    hardware_interface::JointStateInterface joint_state_interface_;
    hardware_interface::VelocityJointInterface velocity_joint_interface_;
    hardware_interface::EffortJointInterface effort_joint_interface_;
    TwistCommandInterface twist_command_interface_;
    // joint_limits_interface::VelocityJointSoftLimitsInterface jnt_velocity_limits_interface_;
    // joint_limits_interface::EffortJointSoftLimitsInterface jnt_effort_limits_interface_;
    JointTrajectoryCommandInterface joint_trajectory_command_interface;

    ros::Subscriber effort_command_sub_; // Subscriber for external effort commands
    void effortCommandCallback(const std_msgs::Float64MultiArray::ConstPtr& msg); // Callback function to update external efforts

    ros::Subscriber acceleration_sub_;
    void jointAccelerationCallback(const std_msgs::Float64MultiArray::ConstPtr& msg);

    std::vector<double> joint_accelerations_;

    /// @brief Joint configuration
    std::vector<std::string> joint_names_;
    std::size_t num_joints_;

    /// @brief Joint state interface types
    std::vector<double> joint_position_;
    std::vector<double> joint_velocity_;
    std::vector<double> joint_effort_;
    geometry_msgs::Twist twist_;

    /// @brief Joint states for setter
    std::vector<double> pos_;
    std::vector<double> vel_;
    std::vector<double> effort_;

    /// @brief Joint state interface types
    std::vector<double> joint_velocity_command_;
    std::vector<double> joint_effort_command_;
    std::vector<int> ef_pose_;

    std::vector<bool> joint_effort_adjusted_;

    /// @brief Joint command for getter
    std::vector<double> velocity_command_;
    std::vector<double> effort_command_;
    std::vector<double> external_effort_command_;
    std::vector<double> last_effort_command_; // Stores the last commanded efforts for each joint


    // Publishers for torque data
    ros::Publisher relative_torque_pub_;
    ros::Publisher gravity_pub_;
    ros::Publisher mass_matrix_pub_;
    ros::Publisher coriolis_pub_;
    ros::Publisher mass_torque_pub_;

    // Friction model instance
    FrictionModel friction_model_;  // Declare the FrictionModel as a private member


    // Messages for publishing
    std_msgs::Float64MultiArray gravity_msg;
    std_msgs::Float64MultiArray mass_matrix_msg;
    std_msgs::Float64MultiArray coriolis_msg;
    std_msgs::Float64MultiArray mass_torque_msg;

    std::chrono::steady_clock::time_point last_acceleration_time_;
    std::chrono::steady_clock::time_point last_effort_time_;

    std::mutex acceleration_mutex_;

    /// @brief frame id of end effector
    std::string frame_id_;

    /// @brief Joint control modes flags
    bool velocity_control_mode_;
    bool effort_control_mode_;
    bool twist_control_mode_;

    /// @brief Joint control mode (ROS param)
    std::string control_mode_;

    /// @brief event status
    EventStatus event_status_;

    KDL::Tree kdl_tree_;
    KDL::Chain kdl_chain_;
    std::shared_ptr<KDL::ChainDynParam> chain_dyn_param_;


    /// @brief Joint control modes flags
    bool linear_velocity_mode_;

    bool valid_feedback_received_;

    /// @brief Joint trajectory messsage for multipoint command interface
    trajectory_msgs::JointTrajectory joint_trajectory_;

    /// @brief Joint trajectory command for getter
    trajectory_msgs::JointTrajectory trajectory_;

    /// @brief command to cancel the current goal
    bool cancel_goal_;

    /// @brief offload trajectory flag
    bool offload_traj_;
    
    void initializeKDL();

    void publishRelativeTorque();

    bool extractPoint_(const trajectory_msgs::JointTrajectory &, std::vector<double> &, double &, const int & = 0);

    void print_(const std::vector<double> &);

    bool getLinearVelocityConfig_(const trajectory_msgs::JointTrajectory &, ControllerConfig &);

    bool getMultiPointConfig_(const trajectory_msgs::JointTrajectory &,ControllerConfig &);
};
}

#endif