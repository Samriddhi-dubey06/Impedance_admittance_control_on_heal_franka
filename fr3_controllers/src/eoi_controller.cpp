#include <fr3_controllers/eoi_controller.h>

#include <cmath>
#include <memory>
#include <deque>
#include <random>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <fr3_controllers/pseudo_inversion.h>

namespace fr3_controllers {

bool EOIController::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) {
    
    std::vector<double> cartesian_stiffness_vector;
    std::vector<double> cartesian_damping_vector;

    std::string arm_id;
    if (!node_handle.getParam("arm_id", arm_id)) {
        ROS_ERROR_STREAM("EOIController: Could not read parameter arm_id");
        return false;
    }

    std::vector<std::string> joint_names;
    if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
        ROS_ERROR(
            "EOIController: Invalid or no joint_names parameters provided, "
            "aborting controller init!");
        return false;
    }

    // Get new parameters for delay and noise.
    // delay_jacobian_ and delay_cartesian_ are specified in number of update cycles.
    // noise_jacobian_ and noise_cartesian_ are standard deviations for the Gaussian noise.
    node_handle.param("delay_jacobian", delay_jacobian_, 10);
    node_handle.param("delay_cartesian", delay_cartesian_, 10);
    node_handle.param("noise_jacobian", noise_jacobian_, 0.0);
    node_handle.param("noise_cartesian", noise_cartesian_, 0.0);

    node_handle.param("K_y", K_y_, 0.0);

    auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
    if (model_interface == nullptr) {
        ROS_ERROR_STREAM("EOIController: Error getting model interface from hardware");
        return false;
    }
    try {
        model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
            model_interface->getHandle(arm_id + "_model"));
    } catch (hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM(
            "EOIController: Exception getting model handle from interface: "
            << ex.what());
        return false;
    }

    auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
    if (state_interface == nullptr) {
        ROS_ERROR_STREAM("EOIController: Error getting state interface from hardware");
        return false;
    }
    try {
        state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
            state_interface->getHandle(arm_id + "_robot"));
    } catch (hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM(
            "EOIController: Exception getting state handle from interface: "
            << ex.what());
        return false;
    }

    auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
    if (effort_joint_interface == nullptr) {
        ROS_ERROR_STREAM("EOIController: Error getting effort joint interface from hardware");
        return false;
    }
    for (size_t i = 0; i < 7; ++i) {
        try {
            joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
        } catch (const hardware_interface::HardwareInterfaceException& ex) {
            ROS_ERROR_STREAM("EOIController: Exception getting joint handles: " << ex.what());
            return false;
        }
    }

    position_d_.setZero();
    orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
    position_d_target_.setZero();
    orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

    cartesian_stiffness_.setZero();
    cartesian_damping_.setZero();

    dq_prev_.setZero();
    ddq_estimate_.setZero();

    // Initialize delay buffers (they are empty at startup).
    jacobian_buffer_.clear();
    error_buffer_.clear();

    return true;
}

void EOIController::starting(const ros::Time& /*time*/) {
    // These are the desired stiffness values along each DOF.
    Eigen::Matrix<double, 6, 6> stiffness;
    stiffness.setZero();
    stiffness(0, 0) = 0.0;      // x translational stiffness
    stiffness(1, 1) = 0.0;      // y translational stiffness
    stiffness(2, 2) = 2000.0;   // z translational stiffness
    stiffness(3, 3) = 50.0;     // roll stiffness
    stiffness(4, 4) = 50.0;     // pitch stiffness
    stiffness(5, 5) = 50.0;     // yaw stiffness

    // For a damping ratio of 1, a common choice is 2*sqrt(stiffness).
    Eigen::Matrix<double, 6, 6> damping;
    damping.setZero();
    damping(0, 0) = 2.0 * sqrt(stiffness(0, 0));
    damping(1, 1) = 2.0 * sqrt(stiffness(1, 1));
    damping(2, 2) = 2.0 * sqrt(stiffness(2, 2));
    damping(3, 3) = 2.0 * sqrt(stiffness(3, 3));
    damping(4, 4) = 2.0 * sqrt(stiffness(4, 4));
    damping(5, 5) = 2.0 * sqrt(stiffness(5, 5));

    // Assign these values to the target impedance.
    cartesian_stiffness_ = stiffness;
    cartesian_damping_ = damping;

    franka::RobotState initial_state = state_handle_->getRobotState();
    std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
    Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
    Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

    // Set Equilibrium Pose to current state
    position_d_ = initial_transform.translation();
    orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());
    position_d_target_ = initial_transform.translation();
    orientation_d_target_ = Eigen::Quaterniond(initial_transform.rotation());

    // Set Nullspace Equilibrium configuration to initial q
    q_d_nullspace_ = q_initial;
    force_ramp_start_time = ros::Time::now(); // Debojit Made this Change 

}

void EOIController::update(const ros::Time& /*time*/, const ros::Duration& period) {
    // Retrieve robot state, coriolis, and Jacobian from the hardware interface.
    franka::RobotState robot_state = state_handle_->getRobotState();
    std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
    std::array<double, 49> mass_array = model_handle_->getMass();
    // NOTE: We no longer use the default Jacobian for τ_force.
    // Instead, we will compute a noise‐affected and delayed Jacobian below.

    // Map the raw arrays to Eigen vectors/matrices.
    Eigen::Matrix<double, 7, 7> mass = Eigen::Map<Eigen::Matrix<double, 7, 7>>(mass_array.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());

    // Get the “current” (undelayed) Jacobian for other control parts.
    std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
    Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());

    // Map robot state.
    Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(robot_state.tau_J_d.data());

    Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
    Eigen::Vector3d position(transform.translation());
    Eigen::Quaterniond orientation(transform.rotation());

    // static bool first_update = true;
    // if (first_update) {
    //     position_d_ = transform.translation();
    //     orientation_d_ = Eigen::Quaterniond(transform.rotation());
    //     position_d_target_ = position_d_;
    //     orientation_d_target_ = orientation_d_;
    //     first_update = false;
    // }

    // Compute the error in Cartesian space (position and orientation).
    Eigen::Matrix<double, 6, 1> error;
    error.head(3) << position - position_d_;
    if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
        orientation.coeffs() << -orientation.coeffs();
    }
    Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
    error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
    error.tail(3) << -transform.rotation() * error.tail(3);

    // Compute impedance control torque: tau_imp = J^T * (-K*error - D*(J*dq))
    Eigen::VectorXd tau_impedance(7);
    Eigen::MatrixXd jacobian_transpose_pinv;
    pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);
    tau_impedance = jacobian.transpose() * (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq));

    // Compute nullspace control torque.
    Eigen::VectorXd tau_nullspace(7);
    tau_nullspace = (Eigen::MatrixXd::Identity(7, 7) - jacobian.transpose() * jacobian_transpose_pinv) *
                    (nullspace_stiffness_ * (q_d_nullspace_ - q) - (2.0 * sqrt(nullspace_stiffness_)) * dq);

    // ========= Begin EOI τ_force modifications =========
    // EOI Tasks: We want to add a delay and noise only in the τ_force part.
    // In particular, we add Gaussian noise to the joint angles (used for Jacobian computation)
    // and to the Cartesian error. We then delay these signals by a fixed number of update cycles.
    
    // --- Compute noise-modified joint angles for Jacobian ---
    static std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<double> noise_jac_dist(0.0, noise_jacobian_);
    std::normal_distribution<double> noise_err_dist(0.0, noise_cartesian_);
    
    std::array<double, 7> q_noisy;
    for (size_t i = 0; i < 7; ++i) {
        q_noisy[i] = q(i) + noise_jac_dist(generator);
    }
    
    // Prepare identity transforms for F_T_EE and EE_T_K.
    // Yaha par consturctor parameter hain! 
    // https://docs.ros.org/en/kinetic/api/franka_hw/html/classfranka__hw_1_1FrankaModelHandle.html#a89802dbc0e58ff40e91b8b7845da92c9
    std::array<double, 16> F_T_EE;
    std::array<double, 16> EE_T_K;
    for (int i = 0; i < 16; ++i) {
        F_T_EE[i] = (i % 5 == 0) ? 1.0 : 0.0;
        EE_T_K[i] = (i % 5 == 0) ? 1.0 : 0.0;
    }
    
    // Compute the noise-affected Jacobian using the new overloaded function.
    std::array<double, 42> jacobian_array_noisy = 
        model_handle_->getZeroJacobian(franka::Frame::kEndEffector, q_noisy, F_T_EE, EE_T_K);
    Eigen::Matrix<double, 6, 7> jacobian_noisy = Eigen::Map<Eigen::Matrix<double, 6, 7>>(jacobian_array_noisy.data());
    
    // --- Compute noise-modified Cartesian error ---
    Eigen::Matrix<double, 6, 1> error_noisy = error;
    for (int i = 0; i < 6; ++i) {
        error_noisy(i) += noise_err_dist(generator);
    }
    
    // --- Delay handling ---
    // Push current noisy signals into their delay buffers.
    jacobian_buffer_.push_back(jacobian_noisy);
    if (jacobian_buffer_.size() > static_cast<size_t>(delay_jacobian_ + 1)) {
        jacobian_buffer_.pop_front();
    }
    error_buffer_.push_back(error_noisy);
    if (error_buffer_.size() > static_cast<size_t>(delay_cartesian_ + 1)) {
        error_buffer_.pop_front();
    }
    
    // Retrieve delayed values if delay is nonzero and sufficient history is available.
    Eigen::Matrix<double, 6, 7> delayed_jacobian;
    if (delay_jacobian_ > 0 && jacobian_buffer_.size() > static_cast<size_t>(delay_jacobian_)) {
        delayed_jacobian = jacobian_buffer_.front();
    } else {
        delayed_jacobian = jacobian_noisy;
    }
    
    Eigen::Matrix<double, 6, 1> delayed_error;
    if (delay_cartesian_ > 0 && error_buffer_.size() > static_cast<size_t>(delay_cartesian_)) {
        delayed_error = error_buffer_.front();
    } else {
        delayed_error = error_noisy;
    }
    
    // --- Recompute force vector for τ_force using the delayed error ---
    Eigen::Matrix<double, 6, 1> F_delay;
    F_delay.setZero();
    // For example, along y (index 1) we use a spring-like force:
    // double K_y = 100.0;  // N/m (as before)
    // F_delay(1) = -K_y * delayed_error(1);
    F_delay(1) = -K_y_ * delayed_error(1);
    // Along x (index 0): a ramping force.
    // static ros::Time force_ramp_start_time = ros::Time::now();
    // double elapsed_force_time = (ros::Time::now() - force_ramp_start_time).toSec();
    static bool ramp_initialized = false;
    // ros::Time force_ramp_start_time;
    if (!ramp_initialized) {
        force_ramp_start_time = ros::Time::now();
        ramp_initialized = true;
    }
    double elapsed_force_time = (ros::Time::now() - force_ramp_start_time).toSec();

    double ramp_duration = 40.0;     // Ramp period in seconds.
    double F_x_min = 1.0;            // Starting force in Newtons.
    double F_x_max = 40.0;           // Final force in Newtons.
    double F_x_desired = F_x_min;
    // if (elapsed_force_time < ramp_duration) {
    //     F_x_desired = F_x_min + (F_x_max - F_x_min) * (elapsed_force_time / ramp_duration);
    // } else {
    //     F_x_desired = F_x_max;
    // }
    if (elapsed_force_time < 0.5) {
        F_x_desired = 0.0;  // no force for first 0.5 sec
    } else if (elapsed_force_time < ramp_duration) {
        F_x_desired = F_x_min + (F_x_max - F_x_min) * ((elapsed_force_time - 0.5) / (ramp_duration - 0.5));
    } else {
        F_x_desired = F_x_max;
    }
    
    F_delay(0) = F_x_desired;
    
    // Compute τ_force using the delayed Jacobian and delayed force.
    Eigen::VectorXd tau_force(7);
    tau_force = delayed_jacobian.transpose() * F_delay;
    // ========= End EOI τ_force modifications =========

    // Compute torque to cancel Inertia.
    double dt = 1e-3;
    if (dq_prev_.isZero(0)) {
        dq_prev_ = dq;
        ddq_estimate_.setZero();
        return;
    }
    Eigen::Matrix<double, 7, 1> ddq_measured = (dq - dq_prev_) / dt;
    ddq_estimate_ = filter_gain * ddq_measured + (1.0 - filter_gain) * ddq_estimate_;
    dq_prev_ = dq;
    Eigen::VectorXd tau_mass(7);
    tau_mass = mass * ddq_estimate_;

    Eigen::VectorXd tau_d(7);
    // Combine torques: note that τ_force now includes the delay and noise effects.
    tau_d << tau_impedance + coriolis + tau_mass + tau_force;
    tau_d << saturateTorqueRate(tau_d, tau_J_d);

    for (size_t i = 0; i < 7; ++i) {
        joint_handles_[i].setCommand(tau_d(i));
    }

    nullspace_stiffness_ = filter_params_ * nullspace_stiffness_target_ +
                           (1.0 - filter_params_) * nullspace_stiffness_;
    position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
    orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
}

Eigen::Matrix<double, 7, 1> EOIController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {
  
    Eigen::Matrix<double, 7, 1> tau_d_saturated{};
    for (size_t i = 0; i < 7; i++) {
        double difference = tau_d_calculated[i] - tau_J_d[i];
        tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
    }
    return tau_d_saturated;
}

}  // namespace fr3_controllers

PLUGINLIB_EXPORT_CLASS(fr3_controllers::EOIController, controller_interface::ControllerBase)
