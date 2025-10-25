#include "addverb_cobot_hw.h"

/**
 * @brief Hardware interface for ROS control
 */

namespace addverb
{   
    void AddverbCobotHw::setControlMode()
    {
        if (control_mode_ == "velocity")
        {
            velocity_control_mode_ = true;
        }
        else if (control_mode_ == "effort")
        {
            effort_control_mode_ = true;
        }
        else if (control_mode_ == "twist")
        {
            twist_control_mode_ = true;
        }
        else if (control_mode_ == "joint_trajectory")
        {
            linear_velocity_mode_ = true;
        }
        else
        {
            ROS_WARN("Selected ros control mode is not valid");
        }
    }

    /**
     * @brief update the local copy of the event execution status
     *
     * @param event
     */
    void AddverbCobotHw::set(const EventStatus &event)
    {
        event_status_ = event;
    }

    /**
     * @brief perform additional computation based on the control mode opted for
     *
     */
    void AddverbCobotHw::performAddCompute()
    {
        if (linear_velocity_mode_)
        {
            if (event_status_.status == static_cast<int>(EventExecutionStatus::eExecuting))
            {
                offload_traj_ = false;
            }
            else if (event_status_.event != static_cast<int>(Event::eDisableRobot))
            {
                offload_traj_ = true;
            }
            // else if (event_status_.event == static_cast<int>(Event::eResetConfig))
            // {
                // offload_traj_ = true;
            // }
        }
    }

    /**
     * @brief update the robot_state to joint_handles
     *
     * @param now
     * @param elapsed_time
     */
    void AddverbCobotHw::read(const ros::Time &now, const ros::Duration &elapsed_time)
    {
        joint_position_ = pos_;
        joint_velocity_ = vel_;
        joint_effort_ = effort_;
    }

    /**
     * @brief update the control command from command_handles to local data members
     *
     * @param now
     * @param elapsed_time
     */

    addverb::AddverbCobotHw::AddverbCobotHw(ros::NodeHandle &nh) 
    : nh_(nh), num_joints_(6), velocity_control_mode_(false),
      effort_control_mode_(false), frame_id_("end_effector"), 
      twist_control_mode_(false), linear_velocity_mode_(false), 
      cancel_goal_(false), friction_model_()
    {
        // ROS joint state interface containers
        joint_position_.resize(num_joints_, 0.0);
        joint_velocity_.resize(num_joints_, 0.0);
        joint_effort_.resize(num_joints_, 0.0);
        ef_pose_.resize(6, 0.0);

        // Intermediate joint state containers
        pos_.resize(num_joints_, 0.0);
        vel_.resize(num_joints_, 0.0);
        effort_.resize(num_joints_, 0.0);
        external_effort_command_.resize(num_joints_, 0.0);  // Safe initialization


        joint_velocity_command_.resize(num_joints_, 0.0);
        joint_effort_command_.resize(num_joints_, 0.0);
        velocity_command_.resize(num_joints_, 0.0);
        effort_command_.resize(num_joints_, 0.0);
        joint_accelerations_.resize(num_joints_, 0.0);

        // Instead of using global "/ros_control_mode"
        while (!nh_.hasParam("ros_control_mode"))
        {
            ROS_WARN("Waiting for ros_control_mode param in local namespace...");
            ros::Duration(0.1).sleep();
        }

        nh_.getParam("ros_control_mode", control_mode_);
        setControlMode();

        // read "robot_namespace" param from the server
        std::string robot_ns;
        if (!nh_.getParam("robot_namespace", robot_ns))
        {
            ROS_WARN("Parameter 'robot_namespace' not set. Using default robot config.");
            robot_ns = "";  // Fallback to default config
        }
        else
        {
            ROS_INFO_STREAM("Using robot namespace: " << robot_ns);
        }

        // Now set the friction constants
        friction_model_.setConstantsForRobot(robot_ns);

        for (int joint_id = 0; joint_id < num_joints_; joint_id++)
        {
            std::stringstream jointName;
            jointName << "Joint" << "_" << joint_id + 1;
            joint_names_.push_back(jointName.str());

            // Joint Limits Interface Setup
            joint_limits_interface::JointLimits limits;
            if (!joint_limits_interface::getJointLimits(joint_names_[joint_id], nh_, limits))
            {
                ROS_WARN("Failed to load joint limits for %s", joint_names_[joint_id].c_str());
            }

            // Register joint states interface
            joint_state_interface_.registerHandle(hardware_interface::JointStateHandle(
                joint_names_[joint_id], &joint_position_[joint_id], &joint_velocity_[joint_id], &joint_effort_[joint_id]));

            // Register joint velocity command interface
            hardware_interface::JointHandle joint_handle_velocity = hardware_interface::JointHandle(
                joint_state_interface_.getHandle(joint_names_[joint_id]), &joint_velocity_command_[joint_id]);
            velocity_joint_interface_.registerHandle(joint_handle_velocity);

            // Register joint effort command interface
            hardware_interface::JointHandle joint_handle_effort = hardware_interface::JointHandle(
                joint_state_interface_.getHandle(joint_names_[joint_id]), &joint_effort_command_[joint_id]);
            effort_joint_interface_.registerHandle(joint_handle_effort);
        }

        twist_command_interface_.registerHandle(TwistCommandHandle(frame_id_, &twist_));
        joint_trajectory_command_interface.registerHandle(JointTrajectoryCommandHandle(control_mode_,
                                                                                                &joint_trajectory_, &cancel_goal_, &offload_traj_));

        // Registering with ROS control hardware interface
        registerInterface(&joint_state_interface_);
        registerInterface(&velocity_joint_interface_);
        registerInterface(&effort_joint_interface_);
        registerInterface(&twist_command_interface_);
        registerInterface(&joint_trajectory_command_interface);

         // Initialize the effort command subscriber
        effort_command_sub_ = nh_.subscribe<std_msgs::Float64MultiArray>(
            "effort_controller/command", 10, &AddverbCobotHw::effortCommandCallback, this);

        acceleration_sub_ = nh_.subscribe<std_msgs::Float64MultiArray>(
            "joint_accelerations", 10, &AddverbCobotHw::jointAccelerationCallback, this);

        // In the constructor
        relative_torque_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("relative_torque", 10);
        gravity_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("gravity_torque", 10);
        mass_matrix_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("mass_matrix", 10);
        coriolis_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("coriolis_torque", 10);
        mass_torque_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("mass_torque", 10);



        ROS_INFO("Subscribed to /effort_controller/command for external effort commands.");

        // Initialize KDL
        initializeKDL();
    }

    void AddverbCobotHw::effortCommandCallback(const std_msgs::Float64MultiArray::ConstPtr& msg)
    {
        if (msg->data.size() != num_joints_)
        {
            ROS_ERROR("Received effort command size %zu does not match number of joints %zu", msg->data.size(), num_joints_);
            return;
        }

        for (size_t i = 0; i < num_joints_; i++)
        {
            external_effort_command_[i] = msg->data[i];
        }

        // Update the timestamp for the last received effort command
        last_effort_time_ = std::chrono::steady_clock::now();

        ROS_DEBUG("Updated external effort command.");
    }


    void AddverbCobotHw::initializeKDL()
    {
        // Load URDF from parameter server
        std::string urdf_string;
        if (!nh_.getParam("robot_description", urdf_string))
        {
            ROS_ERROR("Failed to get robot_description from parameter server");
            return;
        }

        urdf::Model model;
        if (!model.initString(urdf_string))
        {
            ROS_ERROR("Failed to parse URDF string");
            return;
        }

        if (!kdl_parser::treeFromUrdfModel(model, kdl_tree_))
        {
            ROS_ERROR("Failed to construct KDL tree");
            return;
        }

        // Ensure these link names match your URDF links
        std::string base_link = "base_link";
        std::string tip_link = "tool";

        if (!kdl_tree_.getChain(base_link, tip_link, kdl_chain_))
        {
            ROS_ERROR("Failed to get KDL chain from %s to %s", base_link.c_str(), tip_link.c_str());
            return;
        }

        ROS_INFO("Successfully loaded KDL chain from %s to %s", base_link.c_str(), tip_link.c_str());

        KDL::Vector gravity(0.0, 0.0, -9.81);
        chain_dyn_param_ = std::make_shared<KDL::ChainDynParam>(kdl_chain_, gravity);
    }



    void AddverbCobotHw::waitForFirstValidFeedback()
    {
        ROS_INFO("Waiting for first valid joint feedback ...");
        ros::Rate loop_rate(50); // 50 Hz

        while (!valid_feedback_received_ && ros::ok())
        {
            ros::spinOnce(); // process any pending callbacks
            loop_rate.sleep();
        }

        // Once we exit this loop, valid_feedback_received_ == true
        ROS_INFO("Received first valid feedback. Proceeding with initialization.");
    }

    const std::vector<double>& AddverbCobotHw::getEffortCommands() const
    {
        return external_effort_command_;
    }




    void AddverbCobotHw::write(const ros::Time &now, const ros::Duration &elapsed_time)
    {   
        // Now wait for at least one valid RobotFeedback to come in:
        // waitForFirstValidFeedback();
        if (velocity_control_mode_)
        {
            velocity_command_ = joint_velocity_command_;
        }
        
        else if (effort_control_mode_)
        {   

            // Ensure KDL is initialized
            if (!chain_dyn_param_)
            {
                ROS_ERROR_THROTTLE(1.0, "KDL ChainDynParam not initialized. Skipping effort control.");
                return;
            }

             // Reset effort command to zero before applying gravity compensation
            std::fill(joint_effort_command_.begin(), joint_effort_command_.end(), 0.0);

            // Ensure external commands are zeroed out if not explicitly set
            // std::fill(external_effort_command_.begin(), external_effort_command_.end(), 0.0);

            // Compute gravity torques based on current joint positions
            std::vector<double> gravity_torques(num_joints_, 0.0);
            computeGravityTorques(pos_, gravity_torques); //g(q)

            // Compute Mass Matrix (B(q))
            std::vector<std::vector<double>> mass_matrix(num_joints_, std::vector<double>(num_joints_, 0.0));
            computeMassMatrix(pos_, mass_matrix);

            // Compute Coriolis Torques (C(q, q_dot))
            std::vector<double> coriolis_torques(num_joints_, 0.0);
            computeCoriolis(pos_, vel_, coriolis_torques);

            // Placeholder for joint accelerations (assume zero if not provided)
            std::vector<double> joint_accelerations(num_joints_, 0.0);
            
            if (std::chrono::steady_clock::now() - last_acceleration_time_ > std::chrono::milliseconds(100)) {
                ROS_WARN_ONCE("No acceleration data received recently; defaulting to zero accelerations.");
                std::fill(joint_accelerations_.begin(), joint_accelerations_.end(), 0.0);
            }

            if (std::chrono::steady_clock::now() - last_effort_time_ > std::chrono::milliseconds(100)) {
                ROS_WARN_ONCE("No external torque commands received recently; defaulting to zero torques.");
                std::fill(external_effort_command_.begin(), external_effort_command_.end(), 0.0);
            }


            // Compute Mass Matrix Torques: B(q) * joint_accelerations
            std::vector<double> mass_matrix_torques(num_joints_, 0.0);
            for (int i = 0; i < num_joints_; i++) {
                for (int j = 0; j < num_joints_; j++) {
                    mass_matrix_torques[i] += mass_matrix[i][j] * joint_accelerations[j];
                }
            }

            mass_torque_msg.data.clear();

            // Publish Mass Torques
            for (int i = 0; i < num_joints_; i++)
            {
                mass_torque_msg.data.push_back(mass_matrix_torques[i]);
            }
            mass_torque_pub_.publish(mass_torque_msg);
            

            // Add gravity compensation, Coriolis, and mass matrix contributions to the commanded efforts
            for (int joint_id = 0; joint_id < num_joints_; joint_id++)
            {   
                // ROS_INFO_STREAM("Joint " << joint_id << " gravity torque: " << gravity_torques[joint_id]);
                ROS_DEBUG("Before gravity compensation: Effort command[%d] = %f", joint_id, joint_effort_command_[joint_id]);
                // Combine gravity compensation
                joint_effort_command_[joint_id] +=  gravity_torques[joint_id] + 
                                                    coriolis_torques[joint_id] + 
                                                    mass_matrix_torques[joint_id]+
                                                    external_effort_command_[joint_id];

                ROS_DEBUG("After gravity compensation: Effort command[%d] = %f", joint_id, joint_effort_command_[joint_id]);
            }

                // Below is the implementation of the friction modelling
                // Compute friction torques from your friction_model_ and add them

                Eigen::Matrix<double, 6, 1> joint_vel_eigen;
                for (int i = 0; i < num_joints_; i++)
                {
                    joint_vel_eigen(i) = vel_[i]; // copy from std::vector<double> to Eigen
                }

                 // Call the friction model
                Eigen::Matrix<double, 6, 1> friction_torques 
                    = friction_model_.computeFrictionTorques(joint_vel_eigen);

                // Add the friction torques to the existing effort commands
                // for (int i = 0; i < num_joints_; i++)
                // {
                //     joint_effort_command_[i] += friction_torques(i);
                // }

             effort_command_ = joint_effort_command_;
        }

        else if (twist_control_mode_)
        {
            ef_pose_[0] = twist_.linear.x;
            ef_pose_[1] = twist_.linear.y;
            ef_pose_[2] = twist_.linear.z;
            ef_pose_[3] = twist_.angular.x;
            ef_pose_[4] = twist_.angular.y;
            ef_pose_[5] = twist_.angular.z;
        }
        else if (linear_velocity_mode_)
        {
            trajectory_ = joint_trajectory_;
        }
    }

    /**
     * @brief update robot_state to hardware interface
     *
     * @param jointfeedback
     */
    void AddverbCobotHw::setter(RobotFeedback &jointfeedback)
    {
        for (int joint_id = 0; joint_id < num_joints_; joint_id++)
        {
            pos_[joint_id] = jointfeedback.jpos[joint_id];
            vel_[joint_id] = jointfeedback.jvel[joint_id];
            effort_[joint_id] = jointfeedback.jtor[joint_id];
        }
        valid_feedback_received_ = true;
    }

    void AddverbCobotHw::getter(ControllerConfig &cconfig, ControlInterrupt &cinterrupt)
    {
        if (velocity_control_mode_)
        {
            getExternalVelocityConfig(cconfig, cinterrupt);
        }
        else if (effort_control_mode_)
        {
            getExternalEffortConfig(cconfig, cinterrupt);
        }
        else if (twist_control_mode_)
        {
            getTeleopConfig(cinterrupt);
        }
        else if (linear_velocity_mode_)
        {
            getLinearVelocityConfig(cconfig);
        }
    }

    void AddverbCobotHw::getExternalVelocityConfig(ControllerConfig &cconfig, ControlInterrupt &cinterrupt)
    {
        ControlInterrupt c_temp;
        for (int joint_id = 0; joint_id < num_joints_; joint_id++)
        {
            c_temp.ext_ctrl.push_back(velocity_command_[joint_id]);
        }
        cinterrupt = c_temp;
    }

    void AddverbCobotHw::getExternalEffortConfig(ControllerConfig &cconfig, ControlInterrupt &cinterrupt)
    {
        ControlInterrupt c_temp;
        for (int joint_id = 0; joint_id < num_joints_; joint_id++)
        {
            c_temp.ext_ctrl.push_back(effort_command_[joint_id]);
            std::cout<<c_temp.ext_ctrl[joint_id]<<"\t";
        }
        std::cout<<std::endl;
        
        cinterrupt = c_temp;
    }

    void AddverbCobotHw::getTeleopConfig(ControlInterrupt &cinterrupt)
    {
        cinterrupt.cart_jog = ef_pose_;
        // cinterrupt.print();
    }

    void AddverbCobotHw::getLinearVelocityConfig(ControllerConfig &cconfig)
    {
        if (trajectory_.points.size() == 0)
        {
            // ROS_WARN("No trajectory uploaded. Kindly feed in a new trajectory");
            cconfig.target_time = -1;
            return;
        }

        if (trajectory_.points[0].time_from_start == ros::Duration(-1))
        {
            // ROS_WARN("No subsequent trajectory given.Will stay in the last trajectory");
            cconfig.target_time = -1;
            return;
        }

        if (trajectory_.points.size() == 1)
        {
            if (!getLinearVelocityConfig_(trajectory_, cconfig))
            {
                ROS_WARN("Incorrect target position. Number of joints exceeds possible number of joints");
                cconfig.target_time = -1;
                return;
            }
        }
        else
        {
            if (!getMultiPointConfig_(trajectory_, cconfig))
            {
                ROS_WARN("Incorrect target position. Number of joints exceeds possible number of joints");
                cconfig.target_time = -1;
                return;
            }
        }
    }

    /**
     * @brief extract target poistion and time interval at the corresponding id
     *
     * @param traj
     * @param pos
     * @param time
     * @param id
     */
    bool AddverbCobotHw::extractPoint_(const trajectory_msgs::JointTrajectory &traj, std::vector<double> &pos, double &time, const int &id)
    {
        pos.clear();

        if (traj.points[id].positions.size() != N_DOF)
        {
            ROS_WARN("target position must be specified for each joint");
            return false;
        }

        if (traj.points[id].time_from_start.toSec() <= 0)
        {
            ROS_WARN("target time must be greater than minimum allowed time_interval");
            return false;
        }

        for (int i = 0; i < traj.points[id].positions.size(); i++)
        {
            pos.push_back(traj.points[id].positions[i]);
        }

        time = traj.points[id].time_from_start.toSec();

        return true;
    }

    /**
     * @brief
     *
     * @param cconfig
     */
    void AddverbCobotHw::getControllerConfig(ControllerConfig &cconfig)
    {
        if (velocity_control_mode_)
        {
            cconfig.controller = static_cast<int>(API::eExternalVelocityAPI);
        }
        else if (effort_control_mode_)
        {
            cconfig.controller = static_cast<int>(API::eExternalTorqueAPI);
        }
        else if (twist_control_mode_)
        {
            cconfig.controller = static_cast<int>(API::eJogAxisAPI);
        }
        else if (linear_velocity_mode_)
        {
            cconfig.controller = static_cast<int>(API::eLinearVelocityAPI);
        }
    }

    /**
     * @brief
     *
     * @param vec
     */
    void AddverbCobotHw::print_(const std::vector<double> &vec)
    {
        for (int i = 0; i < vec.size(); i++)
        {
            //std::cout << vec[i] << "\t";
        }
        std::cout << std::endl;
    }

    void AddverbCobotHw::computeGravityTorques(const std::vector<double> &joint_positions, std::vector<double> &gravity_torques)
    {   
        if (joint_positions.size() != num_joints_)
        {
            ROS_ERROR("Invalid joint positions size. Expected %lu, got %lu", num_joints_, joint_positions.size());
            return;
        }

        // Create a local copy of the joint positions
        KDL::JntArray q(num_joints_);
        for (int i = 0; i < num_joints_; i++)
        {
            // Modify the 3rd joint (index 2) by multiplying with -1
            if (i == 2) 
            {
                q(i) = 1.0 * joint_positions[i];  // Invert the 3rd joint
            }
            else 
            {
                q(i) = joint_positions[i];
            }
        }

        KDL::JntArray g(num_joints_);
        if (!chain_dyn_param_)
        {
            ROS_ERROR("KDL ChainDynParam is not initialized.");
            return;
        }

        chain_dyn_param_->JntToGravity(q, g);

        // Clear the previous data to prevent accumulation
        gravity_msg.data.clear();

        for (int i = 0; i < num_joints_; i++)
        {
            gravity_torques[i] = g(i);
            gravity_msg.data.push_back(g(i));
            ROS_DEBUG("Joint %d: Gravity torque = %f", i, gravity_torques[i]);
        }
        gravity_pub_.publish(gravity_msg);
    }

    void AddverbCobotHw::computeMassMatrix(const std::vector<double> &joint_positions, std::vector<std::vector<double>> &mass_matrix)
    {
        if (joint_positions.size() != num_joints_)
        {
            ROS_ERROR("Invalid joint positions size. Expected %lu, got %lu", num_joints_, joint_positions.size());
            return;
        }

        // Create a local copy of joint positions
        KDL::JntArray q(num_joints_);
        for (int i = 0; i < num_joints_; i++)
        {
            q(i) = joint_positions[i];
        }

        // Prepare the mass matrix
        KDL::JntSpaceInertiaMatrix H(num_joints_); // Mass matrix in KDL format
        if (!chain_dyn_param_)
        {
            ROS_ERROR("KDL ChainDynParam is not initialized.");
            return;
        }

        // Compute the mass matrix
        int ret = chain_dyn_param_->JntToMass(q, H);
        if (ret != 0)
        {
            ROS_ERROR("Failed to compute mass matrix.");
            return;
        }

        mass_matrix_msg.data.clear();

        // Copy the result to the output format
        mass_matrix.resize(num_joints_, std::vector<double>(num_joints_, 0.0));
        for (int i = 0; i < num_joints_; i++)
        {
            for (int j = 0; j < num_joints_; j++)
            {
                mass_matrix[i][j] = H(i, j);
                mass_matrix_msg.data.push_back(H(i, j));
            }
        }
        
        mass_matrix_pub_.publish(mass_matrix_msg);
        ROS_DEBUG("Mass matrix computed successfully.");
    }


    void AddverbCobotHw::computeCoriolis(const std::vector<double> &joint_positions, const std::vector<double> &joint_velocities, std::vector<double> &coriolis_torques)
    {
        if (joint_positions.size() != num_joints_ || joint_velocities.size() != num_joints_)
        {
            ROS_ERROR("Invalid joint positions or velocities size. Expected %lu, got positions: %lu, velocities: %lu", 
                    num_joints_, joint_positions.size(), joint_velocities.size());
            return;
        }

        // Create local copies of joint positions and velocities
        KDL::JntArray q(num_joints_);
        KDL::JntArray q_dot(num_joints_);
        for (int i = 0; i < num_joints_; i++)
        {
            q(i) = joint_positions[i];
            q_dot(i) = joint_velocities[i];
        }

        // Prepare the output array for Coriolis torques
        KDL::JntArray C(num_joints_);

        if (!chain_dyn_param_)
        {
            ROS_ERROR("KDL ChainDynParam is not initialized.");
            return;
        }

        // Compute the Coriolis torques
        int ret = chain_dyn_param_->JntToCoriolis(q, q_dot, C);
        if (ret != 0)
        {
            ROS_ERROR("Failed to compute Coriolis torques.");
            return;
        }

        coriolis_msg.data.clear();
        // Copy the result to the output format
        coriolis_torques.resize(num_joints_, 0.0);
        for (int i = 0; i < num_joints_; i++)
        {
            coriolis_torques[i] = C(i);
            coriolis_msg.data.push_back(C(i));
        }

        // Publish the Coriolis torques
        coriolis_pub_.publish(coriolis_msg);

        ROS_DEBUG("Coriolis torques computed successfully.");
    }

    void AddverbCobotHw::jointAccelerationCallback(const std_msgs::Float64MultiArray::ConstPtr& msg) 
    {
        if (msg->data.size() != num_joints_) {
            ROS_WARN("Received joint acceleration size %zu does not match number of joints %zu", msg->data.size(), num_joints_);
            return;
        }

        // Update joint accelerations
        for (size_t i = 0; i < num_joints_; i++) {
            joint_accelerations_[i] = msg->data[i];
        }
        // Update the timestamp
        last_acceleration_time_ = std::chrono::steady_clock::now();
    }

    /**
     * @brief get config for linear velocity controller
     *
     * @param config
     * @return true
     * @return false
     */
    bool AddverbCobotHw::getLinearVelocityConfig_(const trajectory_msgs::JointTrajectory &traj, ControllerConfig &config)
    {
        config.controller = static_cast<int>(API::eLinearVelocityAPI);

        return extractPoint_(traj, config.target_pos, config.target_time);
    }

    /**
     * @brief get config for linear velocity controller
     *
     * @param config
     * @return true
     * @return false
     */
    bool AddverbCobotHw::getMultiPointConfig_(const trajectory_msgs::JointTrajectory &traj, ControllerConfig &config)
    {
        config.controller = static_cast<int>(API::eMultiPointAPI);
        config.target_pos_seq.clear();
        config.target_time_seq.clear();

        for (int i = 0; i < traj.points.size(); i++)
        {
            std::vector<double> temp_pos;
            double temp_time;
            if (!extractPoint_(traj, temp_pos, temp_time, i))
            {
                return false;
            }
            config.target_pos_seq.push_back(temp_pos);
            config.target_time_seq.push_back(temp_time);
        }

        return true;
    }
}
