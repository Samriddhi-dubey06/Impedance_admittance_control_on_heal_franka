#pragma once

#include <controller_interface/controller_base.h>
#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <Eigen/Core>

namespace fr3_controllers {

class DOB : public controller_interface::MultiInterfaceController<
                                                franka_hw::FrankaModelInterface,
                                                hardware_interface::EffortJointInterface,
                                                franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  
  void momentumObserverStep(
    const Eigen::Matrix<double, 7, 1>& q,
    const Eigen::Matrix<double, 7, 1>& dq,
    const Eigen::Matrix<double, 7, 1>& tau_measured,
    double dt);


  Eigen::Matrix<double, 7, 1> saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d);

  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  Eigen::Matrix<double, 7, 1> p_hat_;
  Eigen::Matrix<double, 7, 1> tau_ext_hat_;
  Eigen::Matrix<double, 7, 7> kp_;

  const double delta_tau_max_{1.0};

  ros::Publisher tau_ext_pub_;
};

}  // namespace fr3_controllers