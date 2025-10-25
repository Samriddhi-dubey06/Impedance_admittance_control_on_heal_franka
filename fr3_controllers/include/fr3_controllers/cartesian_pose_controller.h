// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <array>
#include <memory>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>

#include <franka_hw/franka_cartesian_command_interface.h>

namespace fr3_controllers {

class CartesianPoseController
    : public controller_interface::MultiInterfaceController<franka_hw::FrankaPoseCartesianInterface,
                                                            franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  franka_hw::FrankaPoseCartesianInterface* cartesian_pose_interface_;
  std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;
  ros::Duration elapsed_time_;
  std::array<double, 16> initial_pose_{};

  std::array<double, 16> pose_a_;
  std::array<double, 16> pose_b_;
  std::array<double, 16> pose_c_;
  std::array<double, 16> pose_d_;
  std::array<double, 16> pose_e_;
  std::array<double, 16> pose_f_;

  ros::Duration t1_;
  ros::Duration t2_;
  ros::Duration t3_;
  ros::Duration t4_;
  ros::Duration t5_;
  ros::Duration t6_;
  int phase_;
  ros::Duration pause_duration_;
  ros::Duration pause_duration_b_;
  ros::Duration pause_duration_d_;
  ros::Duration pause_duration_e_;

  std::array<double, 16> current_pose_after_pause_;  

  // Helpers
  static double quinticTimeScaling(double tau);
  static Eigen::Affine3d arrayToEigenTransform(const std::array<double, 16>& pose);
  static std::array<double, 16> eigenTransformToArray(const Eigen::Affine3d& tf);

};

}  // namespace fr3_controllers
