#include <fr3_controllers/cartesian_impedance_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <fr3_controllers/pseudo_inversion.h>

namespace fr3_controllers {

bool CartesianImpedanceController::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) {
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

    position_d_.setZero();
    orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
    position_d_target_.setZero();
    orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;
    cartesian_stiffness_.setZero();
    cartesian_damping_.setZero();

    return true;
}

void CartesianImpedanceController::starting(const ros::Time& /*time*/) {
    // --- Initial (K0) ---
    Eigen::Matrix<double,6,6> stiffness0, damping0;
    stiffness0.setZero();  damping0.setZero();
    // translational
    stiffness0(0,0)=700; stiffness0(1,1)=700; stiffness0(2,2)=700;
    // rotational
    stiffness0(3,3)=10;   stiffness0(4,4)=10;   stiffness0(5,5)=10;
    for (int i=0; i<6; ++i) {
        damping0(i,i) = 2.0 * std::sqrt(stiffness0(i,i));
    }
    cartesian_stiffness_ = stiffness0;
    cartesian_damping_   = damping0;

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

void CartesianImpedanceController::update(const ros::Time& /*time*/, const ros::Duration& /*period*/) {
    // Retrieve robot state, coriolis, and Jacobian from the hardware interface.

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

    tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) - jacobian.transpose() * jacobian_transpose_pinv) *
                        (nullspace_stiffness_ * (q_d_nullspace_ - q) - (2.0 * sqrt(nullspace_stiffness_)) * dq);

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*-------------------------------PUT YOUR LOGIC HERE------------------------------*/

    // 'jacobian' is the matrix that relates joint velocities to end-effector velocities.
    // 'jacobian.transpose()' maps Cartesian forces back into joint torques.

    // 'cartesian_stiffness_' is a 6x6 matrix that defines how stiff the robot should behave in 3D space (position + orientation) when it gets displaced.
    // 'error' is the position/orientation difference between the desired and actual end-effector pose.

    // 'cartesian_damping_' is a 6x6 matrix that resists motion like a damper/shock absorber.
    // 'dq' gives the joint velocities.
    
    tau_task << jacobian.transpose() *(-cartesian_stiffness_*error -(cartesian_damping_*(jacobian*dq)) );

/*---------------------------DO NOT EDIT ANYTHING BELOW---------------------------*/
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/

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

Eigen::Matrix<double, 7, 1> CartesianImpedanceController::saturateTorqueRate(
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

PLUGINLIB_EXPORT_CLASS(fr3_controllers::CartesianImpedanceController, controller_interface::ControllerBase)