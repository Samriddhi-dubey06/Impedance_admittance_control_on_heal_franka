#pragma once

#include <array>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>

namespace fr3_controllers {

class JointPositionController : public controller_interface::MultiInterfaceController<
                                    hardware_interface::PositionJointInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time& time) override;
  void update(const ros::Time& time, const ros::Duration& period) override;

 private:
  hardware_interface::PositionJointInterface* position_joint_interface_;
  std::vector<hardware_interface::JointHandle> joint_handles_;
  std::array<double, 7> initial_position_{};
  std::array<double, 7> target_position_{};
  ros::Duration elapsed_time_;
  ros::Duration move_duration_;

  double quinticTimeScaling(double tau);
};

}  // namespace fr3_controllers
