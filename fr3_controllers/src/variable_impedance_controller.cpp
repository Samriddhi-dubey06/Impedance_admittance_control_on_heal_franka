#include <fr3_controllers/variable_impedance_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <fr3_controllers/pseudo_inversion.h>

namespace fr3_controllers {

bool VIC::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) {
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

    // Subscribe to /snap
    snap_sub_ = node_handle.subscribe("/snap", 1, &VIC::snapCallback, this);

    position_d_.setZero();
    orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
    position_d_target_.setZero();
    orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;
    cartesian_stiffness_.setZero();
    cartesian_damping_.setZero();

    return true;
}

void VIC::starting(const ros::Time& /*time*/) {
    // --- Initial (K0) ---
    Eigen::Matrix<double,6,6> stiffness0, damping0;
    stiffness0.setZero();  damping0.setZero();
    // translational
    stiffness0(0,0)=6000; stiffness0(1,1)=6000; stiffness0(2,2)=6000;
    // rotational
    stiffness0(3,3)=50;   stiffness0(4,4)=50;   stiffness0(5,5)=50;
    for (int i=0; i<6; ++i) {
        damping0(i,i) = 2.0 * std::sqrt(stiffness0(i,i));
    }
    cartesian_stiffness_ = stiffness0;
    cartesian_damping_   = damping0;
    K0_stiffness_        = stiffness0;
    K0_damping_          = damping0;

    // --- Final (Kf) ---
    Eigen::Matrix<double,6,6> stiffnessf, dampingf;
    stiffnessf.setZero();  dampingf.setZero();
    // translational → 50
    stiffnessf(0,0)=50; stiffnessf(1,1)=50; stiffnessf(2,2)=50;
    // rotational    →  5
    stiffnessf(3,3)=5;  stiffnessf(4,4)=5;  stiffnessf(5,5)=5;
    for (int i=0; i<6; ++i) {
        dampingf(i,i) = 2.0 * std::sqrt(stiffnessf(i,i));
    }
    Kf_stiffness_ = stiffnessf;
    Kf_damping_   = dampingf;

    ramping_.store(false);

    // Set equilibrium to current state
    franka::RobotState initial_state = state_handle_->getRobotState();
    std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
    Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
    Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

    // set equilibrium point to current state
    position_d_ = initial_transform.translation();
    orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());
    position_d_target_ = initial_transform.translation();
    orientation_d_target_ = Eigen::Quaterniond(initial_transform.rotation());

    // set nullspace equilibrium configuration to initial q
    q_d_nullspace_ = q_initial;
}

void VIC::snapCallback(const std_msgs::BoolConstPtr& msg) {
    if (msg->data 
        && !ramping_.load()
        && !has_ramped_.load())
    {   
        ramping_.store(true);
        t_snap_sec_.store(ros::Time::now().toSec());
        has_ramped_.store(true);
    }
}

void VIC::update(const ros::Time& /*time*/, const ros::Duration& /*period*/) {
    // Retrieve robot state, coriolis, and Jacobian from the hardware interface.

    if (ramping_.load()) {
        double now   = ros::Time::now().toSec();
        double dt    = now - t_snap_sec_.load();
        double decay = std::exp(-beta_ * dt);
        cartesian_stiffness_ = 
            Kf_stiffness_ * (1 - decay) +
            K0_stiffness_ * decay;
        cartesian_damping_ =
            Kf_damping_ * (1 - decay) +
            K0_damping_ * decay;
        if (decay < 1e-3) {
            ramping_.store(false);
        }
    }

    franka::RobotState robot_state = state_handle_->getRobotState();
    std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
    std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

    // convert to Eigen
    Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
    Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(robot_state.tau_J_d.data());
    Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
    Eigen::Vector3d position(transform.translation());
    Eigen::Quaterniond orientation(transform.rotation());

    // Compute the error in Cartesian space (position and orientation).
    Eigen::Matrix<double, 6, 1> error;
    error.head(3) << position - position_d_;
    if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
        orientation.coeffs() << -orientation.coeffs();
    }
    Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
    error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
    error.tail(3) << -transform.rotation() * error.tail(3);

    Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);
    Eigen::MatrixXd jacobian_transpose_pinv;
    pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

    tau_task << jacobian.transpose() * (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq));
    tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) - jacobian.transpose() * jacobian_transpose_pinv) *
                        (nullspace_stiffness_ * (q_d_nullspace_ - q) - (2.0 * sqrt(nullspace_stiffness_)) * dq);

    tau_d << tau_task + tau_nullspace + coriolis;

    tau_d << saturateTorqueRate(tau_d, tau_J_d);
    for (size_t i = 0; i < 7; ++i) {
        joint_handles_[i].setCommand(tau_d(i));
    }

    nullspace_stiffness_ = filter_params_ * nullspace_stiffness_target_ +
                           (1.0 - filter_params_) * nullspace_stiffness_;
    position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
    orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
}

Eigen::Matrix<double, 7, 1> VIC::saturateTorqueRate(
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

PLUGINLIB_EXPORT_CLASS(fr3_controllers::VIC, controller_interface::ControllerBase)
