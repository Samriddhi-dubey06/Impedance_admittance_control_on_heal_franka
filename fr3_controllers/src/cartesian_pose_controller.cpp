#include <fr3_controllers/cartesian_pose_controller.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include <controller_interface/controller_base.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <hardware_interface/hardware_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fr3_controllers {

bool CartesianPoseController::init(hardware_interface::RobotHW* robot_hardware,
                                   ros::NodeHandle& node_handle) {
  cartesian_pose_interface_ = robot_hardware->get<franka_hw::FrankaPoseCartesianInterface>();
  if (cartesian_pose_interface_ == nullptr) {
    ROS_ERROR("CartesianPoseController: Could not get Cartesian Pose interface from hardware");
    return false;
  }

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("CartesianPoseController: Could not get parameter arm_id");
    return false;
  }

  try {
    cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
        cartesian_pose_interface_->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM("CartesianPoseController: Exception getting Cartesian handle: " << e.what());
    return false;
  }

  return true;
}

void CartesianPoseController::starting(const ros::Time& /* time */) {
  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
  elapsed_time_ = ros::Duration(0.0);
  phase_ = 0;

  // Define Pose A (column-major)
  // pose_a_ = {-0.05300459444215368, -0.03112258835603854, 0.9981091439484469, 0.0, 0.010053201973987236, -0.9994801710546123, -0.03063146372883896, 0.0, 0.9985436625398355, 0.00841058479196265, 0.05328992461045707, 0.0, 0.37353831357837536, 0.03231082225011088, 0.7443936508834754, 1.0};
  // pose_a_ = {-0.0450371346049489, -0.034157247967839144, 0.9984011742227157, 0.0, -0.03710865800679487, -0.9986683052672631, -0.03584033101138401, 0.0, 0.9982958498816614, -0.03866347486325317, 0.04370963075284297, 0.0, 0.3599019227692286, -0.01609396235268122, 0.7607823482258245, 1.0};
  pose_a_ = {-0.0332331961161328, -0.04533463763579926, 0.9984188955933255, 0.0, -0.03798065797366431, -0.9981918366956957, -0.046588545216813604, 0.0, 0.9987257001842973, -0.039468894198892136, 0.031451266780556486, 0.0, 0.47132275798656603, -0.000979858440219944, 0.7416419626792118, 1.0};
  // Define Pose B as Pose A with offset in y
  pose_b_ = pose_a_;
  // pose_b_[13] += 0.105;

  // Duration for each phase
  t1_ = ros::Duration(4.0);  // move to A
  t2_ = ros::Duration(1.0);   // move to B
}

void CartesianPoseController::update(const ros::Time& /* time */, const ros::Duration& period) {
  elapsed_time_ += period;

  double tau = 0.0;
  Eigen::Affine3d start_tf, target_tf;

  if (phase_ == 0) {
    tau = std::min(elapsed_time_.toSec() / t1_.toSec(), 1.0);
    tau = quinticTimeScaling(tau);
    start_tf = arrayToEigenTransform(initial_pose_);
    target_tf = arrayToEigenTransform(pose_a_);

    if (tau >= 1.0) {
      phase_ = 1;
      elapsed_time_ = ros::Duration(0.0);
    }
  } else if (phase_ == 1) {
    tau = std::min(elapsed_time_.toSec() / t2_.toSec(), 1.0);
    tau = quinticTimeScaling(tau);
    start_tf = arrayToEigenTransform(pose_a_);
    target_tf = arrayToEigenTransform(pose_b_);

    if (tau >= 1.0) {
      phase_ = 2;
    }
  } else {
    cartesian_pose_handle_->setCommand(pose_b_);
    return;
  }

  // Interpolate translation
  Eigen::Vector3d trans = (1 - tau) * start_tf.translation() + tau * target_tf.translation();

  // SLERP for rotation
  Eigen::Quaterniond q_start(start_tf.rotation());
  Eigen::Quaterniond q_target(target_tf.rotation());
  Eigen::Quaterniond q_interp = q_start.slerp(tau, q_target);

  // Combine interpolated translation and rotation
  Eigen::Affine3d interp_tf = Eigen::Affine3d::Identity();
  interp_tf.linear() = q_interp.toRotationMatrix();
  interp_tf.translation() = trans;

  cartesian_pose_handle_->setCommand(eigenTransformToArray(interp_tf));
}

// === Helper: Quintic time scaling ===
double CartesianPoseController::quinticTimeScaling(double tau) {
  return 10 * std::pow(tau, 3) - 15 * std::pow(tau, 4) + 6 * std::pow(tau, 5);
}

// === Helper: Convert std::array → Eigen Transform ===
Eigen::Affine3d CartesianPoseController::arrayToEigenTransform(const std::array<double, 16>& pose) {
  Eigen::Matrix4d mat;
  for (int i = 0; i < 16; ++i) {
    mat(i % 4, i / 4) = pose[i];  // column-major
  }
  return Eigen::Affine3d(mat);
}

// === Helper: Convert Eigen Transform → std::array ===
std::array<double, 16> CartesianPoseController::eigenTransformToArray(const Eigen::Affine3d& tf) {
  std::array<double, 16> pose;
  Eigen::Matrix4d mat = tf.matrix();
  for (int i = 0; i < 16; ++i) {
    pose[i] = mat(i % 4, i / 4);  // row-major to column-major
  }
  return pose;
}

}  // namespace fr3_controllers

PLUGINLIB_EXPORT_CLASS(fr3_controllers::CartesianPoseController,
                       controller_interface::ControllerBase)