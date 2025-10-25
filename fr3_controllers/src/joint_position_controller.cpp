#include <pluginlib/class_list_macros.h>
#include <fr3_controllers/joint_position_controller.h>

#include <cmath>
#include <string>

namespace fr3_controllers {

bool JointPositionController::init(hardware_interface::RobotHW* robot_hardware,
                                   ros::NodeHandle& node_handle) {
  position_joint_interface_ = robot_hardware->get<hardware_interface::PositionJointInterface>();
  if (position_joint_interface_ == nullptr) {
    ROS_ERROR("JointPositionController: Could not get PositionJointInterface from hardware.");
    return false;
  }

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("JointPositionController: Could not get parameter 'arm_id'.");
    return false;
  }

  std::vector<std::string> joint_names = {
      arm_id + "_joint1", arm_id + "_joint2", arm_id + "_joint3",
      arm_id + "_joint4", arm_id + "_joint5", arm_id + "_joint6",
      arm_id + "_joint7"};

  try {
    for (const auto& joint_name : joint_names) {
      joint_handles_.push_back(position_joint_interface_->getHandle(joint_name));
    }
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM("JointPositionController: Exception getting joint handles: " << e.what());
    return false;
  }

  return true;
}

void JointPositionController::starting(const ros::Time& /* time */) {
  elapsed_time_ = ros::Duration(0.0);
  for (size_t i = 0; i < 7; ++i) {
    initial_position_[i] = joint_handles_[i].getPosition();
  }

  // Absolute target joint configuration (can change dynamically if needed)
  target_position_ = {0.10885227225483401, -0.6605660951240033, -0.15190368617502398, -2.3906341351781277, -0.16707174351331505, 3.333843131848353, 0.8889329679327047};

  move_duration_ = ros::Duration(4.0);  // Move in 4 seconds
}

void JointPositionController::update(const ros::Time& /* time */, const ros::Duration& period) {
  elapsed_time_ += period;

  double tau = std::min(elapsed_time_.toSec() / move_duration_.toSec(), 1.0);
  tau = quinticTimeScaling(tau);

  for (size_t i = 0; i < 7; ++i) {
    double pos = (1.0 - tau) * initial_position_[i] + tau * target_position_[i];
    joint_handles_[i].setCommand(pos);
  }
}

double JointPositionController::quinticTimeScaling(double tau) {
  return 10 * std::pow(tau, 3) - 15 * std::pow(tau, 4) + 6 * std::pow(tau, 5);
}

}  // namespace fr3_controllers

PLUGINLIB_EXPORT_CLASS(fr3_controllers::JointPositionController,
                       controller_interface::ControllerBase)
