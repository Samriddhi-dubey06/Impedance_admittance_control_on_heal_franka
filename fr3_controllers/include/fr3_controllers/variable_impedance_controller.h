#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <std_msgs/Bool.h>
#include <Eigen/Dense>

#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

namespace fr3_controllers {

class VIC : public controller_interface::MultiInterfaceController<
                        franka_hw::FrankaModelInterface,
                        hardware_interface::EffortJointInterface,
                        franka_hw::FrankaStateInterface> {
public:
    bool init(hardware_interface::RobotHW* robot_hw,
              ros::NodeHandle& node_handle) override;
    void starting(const ros::Time&) override;
    void update(const ros::Time&, const ros::Duration& period) override;

private:
    void snapCallback(const std_msgs::BoolConstPtr& msg);

    Eigen::Matrix<double, 7, 1> saturateTorqueRate(
        const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
        const Eigen::Matrix<double, 7, 1>& tau_J_d);

    // Handles & interfaces
    std::unique_ptr<franka_hw::FrankaStateHandle>   state_handle_;
    std::unique_ptr<franka_hw::FrankaModelHandle>   model_handle_;
    std::vector<hardware_interface::JointHandle>    joint_handles_;

    // Exponential ramp-down
    std::atomic<bool>        ramping_{false};
    std::atomic<bool>        has_ramped_{false};
    std::atomic<double>      t_snap_sec_{0.0};
    static constexpr double  beta_{100.0};        
    Eigen::Matrix<double,6,6> K0_stiffness_, Kf_stiffness_;
    Eigen::Matrix<double,6,6> K0_damping_,   Kf_damping_;
    ros::Subscriber          snap_sub_;

    // Cartesian impedance
    Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
    Eigen::Matrix<double, 6, 6> cartesian_damping_;

    // Nullspace & pose attractor
    double filter_params_{0.005};
    double nullspace_stiffness_{20.0};
    double nullspace_stiffness_target_{20.0};
    const double delta_tau_max_{3.0};

    Eigen::Matrix<double, 7, 1> q_d_nullspace_;
    Eigen::Vector3d            position_d_;
    Eigen::Quaterniond         orientation_d_;
    Eigen::Vector3d            position_d_target_;
    Eigen::Quaterniond         orientation_d_target_;
};

}  // namespace fr3_controllers