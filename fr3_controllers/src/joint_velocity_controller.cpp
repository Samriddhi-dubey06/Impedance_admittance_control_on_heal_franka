#include <fr3_controllers/joint_velocity_controller.h>

#include <cmath>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace fr3_controllers {

bool JointVelocityController::init(hardware_interface::RobotHW* robot_hardware,
                                          ros::NodeHandle& node_handle) {
  velocity_joint_interface_ = robot_hardware->get<hardware_interface::VelocityJointInterface>();
  if (velocity_joint_interface_ == nullptr) {
    ROS_ERROR(
        "JointVelocityExampleController: Error getting velocity joint interface from hardware!");
    return false;
  }

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("JointVelocityExampleController: Could not get parameter arm_id");
    return false;
  }

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names)) {
    ROS_ERROR("JointVelocityExampleController: Could not parse joint names");
  }
  if (joint_names.size() != 7) {
    ROS_ERROR_STREAM("JointVelocityExampleController: Wrong number of joint names, got "
                     << joint_names.size() << " instead of 7 names!");
    return false;
  }

  velocity_joint_handles_.resize(7);
  for (size_t i = 0; i < 7; ++i) {
    try {
      velocity_joint_handles_[i] = velocity_joint_interface_->getHandle(joint_names[i]);
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "JointVelocityExampleController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  // Initialize the velocity command buffer to zeros
  commanded_velocities_.fill(0.0);

  // Subscribe to joint velocity command topic
  velocity_command_sub_ = node_handle.subscribe(
      "joint_velocity_command", 1, &JointVelocityController::velocityCommandCallback, this);

  auto state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("JointVelocityExampleController: Could not get state interface from hardware");
    return false;
  }

  try {
    auto state_handle = state_interface->getHandle(arm_id + "_robot");

  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM("JointVelocityController: Exception getting state handle: " << e.what());
    return false;
  }

  return true;
}

void JointVelocityController::starting(const ros::Time& /* time */) {
  elapsed_time_ = ros::Duration(0.0);
}

void JointVelocityController::update(const ros::Time& /* time */,
                                            const ros::Duration& period) {
  elapsed_time_ += period;

  // Read current commanded velocities (thread-safe)
  std::array<double, 7> current_vels;
  {
    std::lock_guard<std::mutex> lock(velocity_mutex_);
    current_vels = commanded_velocities_;
  }

  // Send velocity command to each joint
  for (size_t i = 0; i < 7; ++i) {
    velocity_joint_handles_[i].setCommand(current_vels[i]);
  }
}

void JointVelocityController::stopping(const ros::Time& /*time*/) {
  // WARNING: DO NOT SEND ZERO VELOCITIES HERE AS IN CASE OF ABORTING DURING MOTION
  // A JUMP TO ZERO WILL BE COMMANDED PUTTING HIGH LOADS ON THE ROBOT. LET THE DEFAULT
  // BUILT-IN STOPPING BEHAVIOR SLOW DOWN THE ROBOT.
}

// Called when message is received on /joint_velocity_command
void JointVelocityController::velocityCommandCallback(const std_msgs::Float64MultiArray::ConstPtr& msg) {
    if (msg->data.size() == 7) {
        std::lock_guard<std::mutex> lock(velocity_mutex_);
        for (size_t i = 0; i < 7; ++i) {
            commanded_velocities_[i] = msg->data[i];
        }
    } else {
    ROS_WARN_THROTTLE(1.0, "JointVelocityController: Expected 7 joint velocities, got %lu", msg->data.size());
    }   
}

}  // namespace fr3_controllers

PLUGINLIB_EXPORT_CLASS(fr3_controllers::JointVelocityController, controller_interface::ControllerBase)
