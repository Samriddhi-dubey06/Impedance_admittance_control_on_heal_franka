#include <fr3_controllers/dob.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <fr3_controllers/pseudo_inversion.h>

namespace fr3_controllers {

bool DOB::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) {
    std::vector<double> cartesian_stiffness_vector;
    std::vector<double> cartesian_damping_vector;

    std::string arm_id;
    if (!node_handle.getParam("arm_id", arm_id)) {
        ROS_ERROR_STREAM("CartesianImpedanceExampleController: Could not read parameter arm_id");
        return false;
    }

    std::vector<std::string> joint_names;
    if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
        ROS_ERROR(
            "CartesianImpedanceExampleController: Invalid or no joint_names parameters provided, "
            "aborting controller init!");
        return false;
    }

    auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
    if (model_interface == nullptr) {
        ROS_ERROR_STREAM(
            "CartesianImpedanceExampleController: Error getting model interface from hardware");
        return false;
    }
    try {
        model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
            model_interface->getHandle(arm_id + "_model"));
    } catch (hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM(
            "CartesianImpedanceExampleController: Exception getting model handle from interface: "
            << ex.what());
        return false;
    }

    auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
    if (state_interface == nullptr) {
        ROS_ERROR_STREAM(
            "CartesianImpedanceExampleController: Error getting state interface from hardware");
        return false;
    }
    try {
        state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
            state_interface->getHandle(arm_id + "_robot"));
    } catch (hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM(
            "CartesianImpedanceExampleController: Exception getting state handle from interface: "
            << ex.what());
        return false;
    }

    auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
    if (effort_joint_interface == nullptr) {
        ROS_ERROR_STREAM(
            "CartesianImpedanceExampleController: Error getting effort joint interface from hardware");
        return false;
    }
    for (size_t i = 0; i < 7; ++i) {
        try {
        joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
        } catch (const hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM(
            "CartesianImpedanceExampleController: Exception getting joint handles: " << ex.what());
        return false;
        }
    }

    tau_ext_pub_ = node_handle.advertise<std_msgs::Float64MultiArray>(
        "tau_ext", 1);
        
    return true;
}

void DOB::starting(const ros::Time& /*time*/) {
    p_hat_.setZero();
    tau_ext_hat_.setZero();
    kp_.setZero();
    
    for (int i = 0; i < 7; ++i) {
    kp_(i, i) = 2.0; 
    }
}

void DOB::update(const ros::Time& /*time*/, const ros::Duration& period) {
    double dt = period.toSec();

    franka::RobotState rs = state_handle_->getRobotState();
    Eigen::Map<Eigen::Matrix<double, 7, 1>> q(rs.q.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(rs.dq.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J(rs.tau_J.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(rs.tau_J_d.data());

    momentumObserverStep(q, dq, tau_J, dt);

    std::array<double, 7> coriolis_array = model_handle_->getCoriolis();

    Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
    
    Eigen::VectorXd tau_command(7);

    tau_command << coriolis;
    tau_command << saturateTorqueRate(tau_command, tau_J_d);
    for (size_t i = 0; i < 7; ++i) {
        joint_handles_[i].setCommand(tau_command(i));
    }
   
}

void DOB::momentumObserverStep(
    const Eigen::Matrix<double, 7, 1>& q,
    const Eigen::Matrix<double, 7, 1>& dq,
    const Eigen::Matrix<double, 7, 1>& tau_measured,
    double dt) {

    auto mass_arr     = model_handle_->getMass();
    auto coriolis_arr = model_handle_->getCoriolis();
    auto gravity_arr  = model_handle_->getGravity();
    Eigen::Map<const Eigen::Matrix<double, 7, 7>> M(mass_arr.data());
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> C(coriolis_arr.data());
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> G(gravity_arr.data());

    Eigen::Matrix<double, 7, 1> p = M * dq;
    Eigen::Matrix<double,7,1> p_hat_dot = C + tau_measured - G + kp_ * (p - p_hat_);
    p_hat_ = p_hat_ + p_hat_dot * dt;
    Eigen::Matrix<double,7,1> e = p - p_hat_;
    Eigen::Matrix<double,7,1> tau_ext_hat_ = kp_ * e;
    
    std_msgs::Float64MultiArray msg;
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label   = "joint";
    msg.layout.dim[0].size    = 7;
    msg.layout.dim[0].stride  = 7;
    msg.data.resize(7);
    for (int i = 0; i < 7; ++i) {
        msg.data[i] = tau_ext_hat_[i];
    }
    tau_ext_pub_.publish(msg);
}

Eigen::Matrix<double, 7, 1> DOB::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  
    Eigen::Matrix<double, 7, 1> tau_d_saturated{};
    for (size_t i = 0; i < 7; i++) {
        double difference = tau_d_calculated[i] - tau_J_d[i];
        tau_d_saturated[i] =
            tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
    }
    return tau_d_saturated;
}

}  // namespace fr3_controllers

PLUGINLIB_EXPORT_CLASS(fr3_controllers::DOB, controller_interface::ControllerBase)