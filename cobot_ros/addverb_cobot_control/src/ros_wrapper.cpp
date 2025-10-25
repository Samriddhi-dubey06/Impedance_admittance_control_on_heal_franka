#include "ros_wrapper.h"

namespace addverb
{

	RosWrapper::RosWrapper(ros::NodeHandle &nh, std::shared_ptr<DataHandler> &handler)
	{
		nh_ = nh;
		std::string robot_ns;
		if (!nh_.getParam("robot_namespace", robot_ns)) {
			ROS_WARN("Parameter 'robot_namespace' not set. Using global namespace.");
			robot_ns = "";
		}
		else {
			ROS_INFO_STREAM("Using robot namespace: " << robot_ns);
		}

		// Ensure it ends with a '/'
		if (!robot_ns.empty() && robot_ns.back() != '/') {
			robot_ns += "/";
		}

		handler_ = handler;

		std::string control_mode;

		// update payload configuration to robot
		// details of failure are handled internally
		if (!updatePayload_())
		{
			ros::shutdown();
			return;
		}

		ros::Time start_time = ros::Time::now();
		ros::Duration timeout(2.0);

		while (ros::Time::now() - start_time < timeout)
		{

		if (nh_.hasParam("ros_control_mode"))
			{
				ROS_INFO("ros_control_mode param loaded in namespace");
				nh_.getParam("ros_control_mode", control_mode);
				break;
			}

			ROS_INFO_THROTTLE(1, "Waiting for ros_control_mode param");
			ros::Duration(0.2).sleep(); // short sleep to avoid spinning too tight
		}

		if (nh_.hasParam("ros_control_mode"))
		{
			nh_.getParam("ros_control_mode", control_mode);
		}
		else
		{
			ROS_ERROR("Timed out while waiting for ros_control_mode param in this namespace");
			ros::shutdown();
			return;
		}

		if (!updateAPI_(control_mode))
		{
			ROS_ERROR("Unknown type of controller");
			return;
		}

		getSafetyMode_();

		// Reset hardware interface
		hw_.reset(new AddverbCobotHw(nh_));

		// Reset control manager
		cm_.reset(new controller_manager::ControllerManager(
			hw_.get(), nh_));

		// go to motion state
		if (!goToMotionState_())
		{
			ROS_ERROR("Failed to start robot");
			return;
		}

		if (shouldResetController_())
		{
			if (hw_->isEffortControlMode()) // Check if the hardware is in Effort Control Mode
			{
				ros::Rate wait_rate(10); // 10 Hz waiting loop
				bool valid_effort_detected = false;
				ros::Time start_time = ros::Time::now();
				ros::Duration timeout(10.0); // 5 seconds timeout

				ROS_INFO("Effort Control Mode detected. Waiting for valid effort commands before resetting the controller...");

				while (ros::ok() && !valid_effort_detected && (ros::Time::now() - start_time) < timeout)
				{
					const std::vector<double>& effort_commands = hw_->getEffortCommands();

					valid_effort_detected = std::any_of(
						effort_commands.begin(),
						effort_commands.end(),
						[](double effort) { return std::abs(effort) > 1e-3; } // Threshold for valid effort
					);

					if (!valid_effort_detected)
					{
						ROS_WARN_THROTTLE(1.0, "Effort commands are still zero. Waiting...");
						wait_rate.sleep();
					}
				}

				if (!valid_effort_detected)
				{
					ROS_ERROR("Timeout waiting for valid effort commands. Proceeding with controller reset cautiously.");
				}
				else
				{
					ROS_INFO("Valid effort commands detected. Proceeding with controller reset.");
				}
			}
			else
			{
				ROS_INFO("Non-effort control mode detected. Proceeding with controller reset directly.");
			}

			resetController_();
		}


		ROS_INFO("Robot started");

		if (api_ == API::eLinearVelocityAPI)
		{
			run_ = std::bind(&RosWrapper::trajControl_, this);
		}
		else
		{
			run_ = std::bind(&RosWrapper::rosControl_, this);
		}

		// Advertise namespaced shutdown service
		shutdown_robot_srv_ = nh_.advertiseService(robot_ns + "shutdown_robot_srv", &RosWrapper::shutdownRobotCB_, this);

		// Clear any lingering config data before control loop
		handler_->clearData(UIDataType::eConfigData);

		// Initialize namespaced action servers with unique_ptr
		err_recovery_as_ = std::make_unique<actionlib::ActionServer<addverb_cobot_msgs::ErrorRecoveryAction>>(
			nh_, robot_ns + "error_recovery",
			std::bind(&RosWrapper::errRecoveryGoalCB_, this, std::placeholders::_1),
			std::bind(&RosWrapper::errRecoveryCancelCB_, this, std::placeholders::_1),
			false);

		grasp_as_ = std::make_unique<actionlib::ActionServer<addverb_cobot_msgs::GraspAction>>(
			nh_, robot_ns + "grasp_action",
			std::bind(&RosWrapper::graspGoalCB_, this, std::placeholders::_1),
			std::bind(&RosWrapper::graspCancelCB_, this, std::placeholders::_1),
			false);

		release_as_ = std::make_unique<actionlib::ActionServer<addverb_cobot_msgs::ReleaseAction>>(
			nh_, robot_ns + "release_action",
			std::bind(&RosWrapper::releaseGoalCB_, this, std::placeholders::_1),
			std::bind(&RosWrapper::releaseCancelCB_, this, std::placeholders::_1),
			false);

		// Start the action servers
		err_recovery_as_->start();
		grasp_as_->start();
		release_as_->start();

		// Launch main control loop
		controlLoop_();

	}


	/**
	 * @brief main control loop
	 *
	 */
	void RosWrapper::controlLoop_()
	{
		ros::Rate rate(loop_hz_);
		clock_gettime(CLOCK_MONOTONIC, &last_time_);
		static const double BILLION = 1000000000.0;

		// control loop
		while (ros::ok())
		{
			
			if (should_shutdown_)
			{
				ros::shutdown();
				return;
			}

			if (!handler_->isConnected())
			{
				ROS_WARN("Lost connection with robot. System will shutdown now.");
				ros::shutdown();
				break;
			}

			mut_.lock();
			handler_->readData();
			mut_.unlock();

			// Get change in time
			clock_gettime(CLOCK_MONOTONIC, &current_time_);

			elapsed_time_ = ros::Duration(current_time_.tv_sec - last_time_.tv_sec +
										  (current_time_.tv_nsec - last_time_.tv_nsec) / BILLION);
			last_time_ = current_time_;

			now_ = ros::Time::now();

			mut_.lock();
			handler_->getRobotFeedback(robot_fdbk_);
			handler_->getState(state_);
			mut_.unlock();

			hw_->setter(robot_fdbk_);

			hw_->read(now_, elapsed_time_);

			checkForError_();

			if (!has_error_.load())
			{
				run_();
			}
			// else action server shall have write permissions

			rate.sleep();
		}
	}

	/**
	 * @brief update error_flag based on robot_state
	 *
	 */
	void RosWrapper::checkForError_()
	{
		// std::cout << "entered check for error\n";

		if (state_ == RobotState::eError)
		{
			while (!has_error_.load())
			{
				has_error_.store(true);
				std::this_thread::sleep_for(std::chrono::nanoseconds(10));

				if (has_error_.load())
				{
					ROS_WARN("Robot gone into error. Recover using ErrorRecoveryAction");
				}
			}
		}

		// std::cout << "exited check for error\n";
	}

	/**
	 * @brief control loop specific to ROS controllers
	 *
	 */
	void RosWrapper::rosControl_()
	{
		cm_->update(now_, elapsed_time_);

		hw_->write(now_, elapsed_time_);

		ControllerConfig conf;
		hw_->getter(conf, c_interrupt_);

		mut_.lock();
		handler_->setUserInput(c_interrupt_);
		handler_->writeData();
		mut_.unlock();
	}

	/**
	 * @brief control loop specific for trajectory execution using ROS controllers
	 *
	 */
	void RosWrapper::trajControl_()
	{
		EventStatus status;
		ControllerConfig config;

		mut_.lock();
		handler_->getEventStatus(status);
		mut_.unlock();

		hw_->set(status);
		hw_->performAddCompute();

		cm_->update(now_, elapsed_time_);
		hw_->write(now_, elapsed_time_);

		hw_->getLinearVelocityConfig(config);

		if (config.target_time == -1)
		{
			// ROS_WARN("Provide new trajectory");
			return;
		}

		// ROS_INFO("Got new traj");
		EventConfig event;
		event.event = static_cast<int>(Event::eResetConfig);

		// std::cout << "calling update of cm\n";

		mut_.lock();
		handler_->appendConfig(config);
		handler_->setUserConfig();
		handler_->setEventConfig(event);
		handler_->writeData();
		mut_.unlock();

		std::this_thread::sleep_for(std::chrono::milliseconds(10));

		mut_.lock();
		handler_->readData();
		handler_->getEventStatus(status);
		mut_.unlock();

		while (status.status != static_cast<int>(EventExecutionStatus::eExecuting))
		{
			if (!ros::ok())
			{
				ROS_WARN("Exiting while updating next trajectory to robot");
				return;
			}
			mut_.lock();
			handler_->writeData();
			mut_.unlock();

			std::this_thread::sleep_for(std::chrono::nanoseconds(10));

			mut_.lock();
			handler_->readData();
			handler_->getEventStatus(status);
			mut_.unlock();
		}

		mut_.lock();
		handler_->clearData(UIDataType::eConfigData);
		handler_->clearData(UIDataType::eEventData);
		handler_->writeData();
		mut_.unlock();

		ROS_INFO("Next trajectory uploaded to robot");
	}

	/**
	 * @brief goal execution callback
	 *
	 * @param gh
	 */
	void RosWrapper::errRecoveryGoalCB_(actionlib::ServerGoalHandle<addverb_cobot_msgs::ErrorRecoveryAction> gh)
	{
		if (!has_error_.load())
		{
			ROS_WARN("The system does not need to recover from error as it is error-free currently.");
			return;
		}

		// reset to base

		gh.setAccepted();

		mut_.lock();
		handler_->setAction(RobotAction::eGoToBaseState);
		handler_->writeData();
		mut_.unlock();

		RobotState state = RobotState::eError;

		while (state == RobotState::eError)
		{
			if (!ros::ok())
			{
				ROS_WARN("Exiting while trying to recover from error");
				return;
			}

			mut_.lock();
			handler_->setAction(RobotAction::eGoToBaseState);
			handler_->writeData();

			std::this_thread::sleep_for(std::chrono::milliseconds(1));

			handler_->readData();
			handler_->getState(state);
			mut_.unlock();
		}

		if (state == RobotState::eMotion)
		{
			ROS_WARN("Failed to recover from error.Press E-stop");
			return;
		}

		ROS_INFO("Updated the robot state");

		std::this_thread::sleep_for(std::chrono::milliseconds(10));

		// reset in motion state
		ControllerConfig cconf;
		cconf.controller = static_cast<int>(API::eExternalVelocityAPI);
		cconf.safety_mode = 0;
		mut_.lock();
		if (has_payload_)
		{
			handler_->appendConfig(pconf_);
		}
		handler_->appendConfig(cconf);
		handler_->setUserConfig();

		handler_->setAction(RobotAction::eGoToMotionState);
		handler_->writeData();
		mut_.unlock();

		while (state == RobotState::eBase)
		{
			if (!ros::ok())
			{
				ROS_WARN("Exiting while trying to recover from error");
				return;
			}

			mut_.lock();
			handler_->setAction(RobotAction::eGoToMotionState);
			handler_->writeData();

			std::this_thread::sleep_for(std::chrono::milliseconds(1));

			handler_->readData();
			handler_->getState(state);
			mut_.unlock();
		}

		if (state == RobotState::eError)
		{
			ROS_WARN("Failed to recover from error.Press E-stop");
			return;
		}

		ROS_INFO("Robot restarted in default mode");

		// reset to home

		mut_.lock();
		handler_->clearData(UIDataType::eConfigData);
		mut_.unlock();

		cconf.controller = static_cast<int>(API::eLinearVelocityAPI);
		cconf.target_pos = std::vector<double>(6, 0);
		cconf.target_time = 15;

		EventStatus event_stat;
		EventConfig event;
		event.event = static_cast<int>(Event::eResetConfig);

		mut_.lock();
		handler_->setEventConfig(event);
		handler_->appendConfig(cconf);
		handler_->setUserConfig();

		handler_->writeData();

		handler_->readData();
		handler_->getEventStatus(event_stat);
		mut_.unlock();

		while (event_stat.status != static_cast<int>(EventExecutionStatus::eExecuting))
		{
			if (!ros::ok())
			{
				ROS_WARN("Exiting while trying to recover from error");
				return;
			}

			mut_.lock();
			handler_->writeData();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			handler_->readData();
			handler_->getEventStatus(event_stat);
			mut_.unlock();
		}

		mut_.lock();
		handler_->clearData(UIDataType::eConfigData);
		handler_->clearData(UIDataType::eEventData);
		mut_.unlock();

		while (event_stat.status == static_cast<int>(EventExecutionStatus::eExecuting))
		{
			if (!ros::ok())
			{
				ROS_WARN("Exiting while trying to recover from error");
				return;
			}

			mut_.lock();
			handler_->writeData();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			handler_->readData();
			handler_->getEventStatus(event_stat);
			mut_.unlock();
		}

		if (event_stat.status == static_cast<int>(EventExecutionStatus::eSucceded))
		{
			ROS_INFO("Robot reset to home position");
		}

		ROS_INFO("Will reset robot to the user-provided configuration. Please wait...");
		// reset in the external velocity mode

		mut_.lock();
		handler_->setAction(RobotAction::eGoToBaseState);
		handler_->writeData();
		mut_.unlock();

		state = RobotState::eMotion;

		while (state == RobotState::eMotion)
		{
			if (!ros::ok())
			{
				ROS_WARN("Exiting while trying to recover from error");
				return;
			}

			mut_.lock();
			handler_->setAction(RobotAction::eGoToBaseState);
			handler_->writeData();

			std::this_thread::sleep_for(std::chrono::milliseconds(1));

			handler_->readData();
			handler_->getState(state);
			mut_.unlock();
		}

		if (state == RobotState::eError)
		{
			ROS_WARN("Got into error.Failed to recover");
			return;
		}

		std::this_thread::sleep_for(std::chrono::seconds(1));

		mut_.lock();
		if (has_payload_)
		{
			handler_->appendConfig(pconf_);
		}
		handler_->appendConfig(cconf_);
		handler_->setUserConfig();

		handler_->setAction(RobotAction::eGoToMotionState);
		handler_->writeData();
		mut_.unlock();

		while (state == RobotState::eBase)
		{
			if (!ros::ok())
			{
				ROS_WARN("Exiting while trying to recover from error");
				return;
			}

			mut_.lock();
			handler_->setAction(RobotAction::eGoToMotionState);
			handler_->writeData();

			std::this_thread::sleep_for(std::chrono::milliseconds(1));

			handler_->readData();
			handler_->getState(state);
			mut_.unlock();
		}

		if (state == RobotState::eError)
		{
			ROS_WARN("Failed to recover from error.Press E-stop");
			return;
		}

		// reset controller to the one that was running before error occurred has to be ahndled by respective controllers

		if (shouldResetController_())
		{
			resetController_();
		}

		while (has_error_.load())
		{
			has_error_.store(false);
			std::this_thread::sleep_for(std::chrono::nanoseconds(10));
		}

		gh.setSucceeded();

		ROS_INFO("Error recovery successful.");
	}

	/**
	 * @brief cancel goal execution callback
	 *
	 * @param gh
	 */
	void RosWrapper::errRecoveryCancelCB_(actionlib::ServerGoalHandle<addverb_cobot_msgs::ErrorRecoveryAction> gh)
	{
		ROS_WARN("You are attempting to abort the action of error recovery. Permission denied");
	}

	/**
	 * @brief cancel goal execution callback
	 *
	 * @param gh
	 */
	void RosWrapper::graspCancelCB_(actionlib::ServerGoalHandle<addverb_cobot_msgs::GraspAction> gh)
	{
		ROS_WARN("It is not possible to cancel the grasp action under process");
	}

	/**
	 * @brief cancel goal execution callback
	 *
	 * @param gh
	 */
	void RosWrapper::releaseCancelCB_(actionlib::ServerGoalHandle<addverb_cobot_msgs::ReleaseAction> gh)
	{
		ROS_WARN("It is not possible to cancel the release action under process");
	}

	/**
	 * @brief goal execution callback
	 *
	 * @param gh
	 */
	void RosWrapper::releaseGoalCB_(actionlib::ServerGoalHandle<addverb_cobot_msgs::ReleaseAction> gh)
	{

		if (gripper_status_.load() == static_cast<int>(Event::eOpenGripper))
		{
			ROS_WARN("Gripper is already in release state");
			return;
		}
		gripper_status_.store(static_cast<int>(Event::eOpenGripper));

		if (has_error_.load())
		{
			ROS_WARN("Cannot release before error is cleared. Call error_recovery_action to recover from the error");
			return;
		}

		if (pconf_.gripper_type < 1 || pconf_.gripper_type > 4)
		{
			ROS_WARN("No gripper attached. Attach a gripper and then try to execute release_action");
			return;
		}

		gh.setAccepted();

		EventConfig event;
		event.event = static_cast<int>(Event::eOpenGripper);

		mut_.lock();
		handler_->setEventConfig(event);
		handler_->writeData();
		mut_.unlock();

		// poll the current event status
		EventStatus status;
		status.status = 0;

		ROS_INFO("Updating event to robot...");

		while (status.status != static_cast<int>(EventExecutionStatus::eExecuting))
		{
			if (!ros::ok())
			{
				ROS_WARN("Interrupted execution of event");
				return;
			}

			mut_.lock();
			handler_->writeData();
			mut_.unlock();

			std::this_thread::sleep_for(std::chrono::milliseconds(10));

			mut_.lock();
			handler_->readData();
			handler_->getEventStatus(status);
			mut_.unlock();
		}

		addverb_cobot_msgs::ReleaseFeedback fdbk;
		fdbk.state = status.status;
		gh.publishFeedback(fdbk);

		ROS_INFO("Releasing...");

		mut_.lock();
		handler_->clearData(UIDataType::eEventData);
		mut_.unlock();

		// waiting till the event status is executing
		while (status.status == static_cast<int>(EventExecutionStatus::eExecuting))
		{
			if (!ros::ok())
			{
				ROS_WARN("Interrupted execution of event");
				return;
			}
			mut_.lock();
			handler_->writeData();
			mut_.unlock();

			std::this_thread::sleep_for(std::chrono::milliseconds(10));

			mut_.lock();
			handler_->readData();
			handler_->getEventStatus(status);
			mut_.unlock();
		}

		// update action client with gripper data
		if (status.status == static_cast<int>(EventExecutionStatus::eSucceded))
		{
			addverb_cobot_msgs::ReleaseResult res;
			res.error_code = 0;
			res.error_msg = "";
			res.state = status.status;
			gh.setSucceeded(res);
			ROS_INFO("Released");
		}
		else
		{
			ROS_WARN("Failed to execute action");
		}
	}

	/**
	 * @brief goal execution callback
	 *
	 * @param gh
	 */
	void RosWrapper::graspGoalCB_(actionlib::ServerGoalHandle<addverb_cobot_msgs::GraspAction> gh)
	{

		if (gripper_status_.load() == static_cast<int>(Event::eCloseGripper))
		{
			ROS_WARN("Gripper is already in grasp state");
			return;
		}

		gripper_status_.store(static_cast<int>(Event::eCloseGripper));

		if (has_error_.load())
		{
			ROS_WARN("Cannot grasp before error is cleared. Call error_recovery_action to recover from the error");
			return;
		}

		if (pconf_.gripper_type < 1 || pconf_.gripper_type > 4)
		{
			ROS_WARN("No gripper attached. Attach a gripper and then try to execute grasp_action");
			return;
		}

		gh.setAccepted();

		actionlib::ActionServer<addverb_cobot_msgs::GraspAction>::Goal goal =
			*gh.getGoal();

		if (goal.grasp_force < 0 || goal.grasp_force > 150)
		{
			ROS_WARN("Invalid grasp force");
			gh.setRejected();
			return;
		}

		if (goal.grasp_force == 0)
		{
			ROS_WARN("Given grasp force = 0. Gripper will not clamp.");
		}

		EventConfig event;
		event.event = static_cast<int>(Event::eCloseGripper);
		event.clamp_force = goal.grasp_force;

		mut_.lock();
		handler_->setEventConfig(event);
		handler_->writeData();
		mut_.unlock();

		// poll the current event status
		EventStatus status;
		status.status = 0;

		ROS_INFO("Updating event to robot...");
		while (status.status != static_cast<int>(EventExecutionStatus::eExecuting))
		{
			if (!ros::ok())
			{
				ROS_WARN("Interrupted execution of event");
				return;
			}

			mut_.lock();
			handler_->writeData();
			mut_.unlock();

			std::this_thread::sleep_for(std::chrono::milliseconds(10));

			mut_.lock();
			handler_->readData();
			handler_->getEventStatus(status);
			mut_.unlock();
		}

		addverb_cobot_msgs::GraspFeedback fdbk;
		fdbk.state = status.status;
		gh.publishFeedback(fdbk);

		ROS_INFO("Grasping..");

		mut_.lock();
		handler_->clearData(UIDataType::eEventData);
		mut_.unlock();

		// waiting till the evnt status is executing
		while (status.status == static_cast<int>(EventExecutionStatus::eExecuting))
		{
			if (!ros::ok())
			{
				ROS_WARN("Interrupted execution of event");
				return;
			}
			mut_.lock();
			handler_->writeData();
			mut_.unlock();

			std::this_thread::sleep_for(std::chrono::milliseconds(10));

			mut_.lock();
			handler_->readData();
			handler_->getEventStatus(status);
			mut_.unlock();
		}

		// update action client with gripper data
		if (status.status == static_cast<int>(EventExecutionStatus::eSucceded))
		{
			addverb_cobot_msgs::GraspResult res;
			res.error_code = 0;
			res.error_msg = "";
			res.state = status.status;
			gh.setSucceeded(res);
			ROS_INFO("Grasped");
		}
		else
		{
			ROS_WARN("Failed to execute action");
		}
	}

	/**
	 * @brief perform transition to motion state of the robot
	 *
	 */
	bool RosWrapper::goToMotionState_()
	{
		cconf_.controller = static_cast<int>(API::eExternalVelocityAPI);
		cconf_.safety_mode = controller_safety_mode_map_[api_];
		handler_->appendConfig(cconf_);
		handler_->setUserConfig();
		handler_->setAction(RobotAction::eGoToMotionState);
		handler_->writeData();

		RobotState state = RobotState::eBase;

		while ((state == RobotState::eBase))
		{
			if (!(ros::ok()))
			{
				ROS_WARN("Exited while trying to start robot");
				return false;
			}
			handler_->setAction(RobotAction::eGoToMotionState);
			handler_->writeData();
			std::this_thread::sleep_for(std::chrono::nanoseconds(10));
			handler_->readData();
			handler_->getState(state);
			// std::cout << "state : " << static_cast<int>(state) << std::endl;
		}

		if (state == RobotState::eError)
		{
			std::cout << "landed in error state\n";
			return false;
		}
		else if (state == RobotState::eBase)
		{
			std::cout << "had to exit coz of ctrl-c\n";
			return false;
		}

		return true;
	}

	/**
	 * @brief update the payload configuration to the robot, if payload is attached to the robot end-effector
	 *
	 * @return true
	 * @return false
	 */
	bool RosWrapper::updatePayload_()
	{
		ROS_INFO("Will Load Payload configuration from server...");

		// Waiting for payload params to load
		// todo : put a timer here and exit with error if time lapses

		ros::Time start_time = ros::Time::now();
		ros::Duration timeout(2.0);

		while (ros::Time::now() - start_time < timeout)
		{
			if (nh_.hasParam("payload_attached"))
			{
				ROS_INFO("Loaded Payload Configuration");
				nh_.getParam("payload_attached", has_payload_);
				break;  // <-- break early
			}

			ROS_INFO_THROTTLE(1, "Waiting for payload_attached parameter in the YAML file");
			ros::Duration(0.2).sleep();
		}

		if (nh_.hasParam("payload_attached"))
		{
			nh_.getParam("payload_attached", has_payload_);
		}
		else
		{
			ROS_ERROR("Request to Load Payload Configuration timed out. Please ensure that Payload configuration files are present with the relevant data");
			return false;
		}

		if (has_payload_)
		{
			getPayload_(pconf_);

			if (!isValidPayload_(pconf_))
			{
				return false;
			}

			handler_->appendConfig(pconf_);
			ROS_INFO("Payload configuration loaded successfully");
		}
		else
		{
			ROS_WARN("Payload configuration for the attachment at the end-effector not specified. Ensure that there is no attachment at end-effector for safe operation");
		}

		return true;
	}

	/**
	 * @brief check for validity of the payload
	 *
	 * @param config
	 * @return true
	 * @return false
	 */
	bool RosWrapper::isValidPayload_(const PayloadConfig &config)
	{
		if ((config.mass < safety_limits::mass_ll) || (config.mass > safety_limits::mass_ul))
		{
			ROS_ERROR("mass provided must be within specified limits");
			return false;
		}

		for (int i = 0; i < 3; i++)
		{
			if (config.moi[i] < 0)
			{
				ROS_ERROR("Principal moment of inertia must have positive values");
				return false;
			}
		}

		// todo : update this to take limit from gripper_config.h files
		if ((config.gripper_type < 0) || (config.gripper_type > 4))
		{
			ROS_ERROR("Unknown type of Gripper");
			return false;
		}

		return true;
	}

	/**
	 * @brief get the payload configuration from the parameter file
	 *
	 */
	void RosWrapper::getPayload_(PayloadConfig &config)
	{
		int gripper_type;
		double mass, comx, comy, comz, ixx, iyy, izz, ixy, ixz, iyz;
		nh_.getParam("mass", mass);
		config.mass = mass;

		nh_.getParam("gripper_type", gripper_type);
		config.gripper_type = gripper_type;

		nh_.getParam("comx", comx);
		nh_.getParam("comy", comy);
		nh_.getParam("comz", comz);
		config.com.push_back(comx);
		config.com.push_back(comy);
		config.com.push_back(comz);

		nh_.getParam("Ixx", ixx);
		nh_.getParam("Iyy", iyy);
		nh_.getParam("Izz", izz);
		nh_.getParam("Ixy", ixy);
		nh_.getParam("Ixz", ixz);
		nh_.getParam("Iyz", iyz);

		config.moi.push_back(ixx);
		config.moi.push_back(iyy);
		config.moi.push_back(izz);
		config.moi.push_back(ixy);
		config.moi.push_back(ixz);
		config.moi.push_back(iyz);

		ROS_INFO("Payload details you provided");
		config.print();
	}

	/**
	 * @brief stop and shutdown robot
	 * this is what the shutdown_service provides
	 *
	 * @return true
	 * @return false
	 */
	bool RosWrapper::shutdown()
	{
		mut_.lock();
		handler_->clearData(UIDataType::eEventData);
		handler_->clearData(UIDataType::eConfigData);
		handler_->writeData();
		mut_.unlock();

		std::this_thread::sleep_for(std::chrono::nanoseconds(10));

		mut_.lock();
		handler_->readData();
		handler_->getState(state_);
		mut_.unlock();

		int count = 0;
		bool connected = true;

		ROS_INFO("Shutting down the robot. Please wait....");

		// wait for robot to reach the base state
		while (state_ != RobotState::eBase && (count <= std::numeric_limits<int>::max() / 2))
		{
			if (has_error_.load())
			{
				ROS_WARN("Robot is in error. Recover from the error using error_recovery_action and then attempt shutdown");
				return false;
			}

			if (!ros::ok())
			{
				ROS_ERROR("Received request to abort the shutdown process before completion. Exiting");
				return false;
			}

			mut_.lock();
			if (!handler_->isConnected())
			{
				connected = false;
			}
			handler_->setAction(RobotAction::eGoToBaseState);
			handler_->writeData();
			mut_.unlock();

			if (!connected)
			{
				break;
			}

			std::this_thread::sleep_for(std::chrono::nanoseconds(10));

			mut_.lock();
			handler_->readData();
			handler_->getState(state_);
			mut_.unlock();

			count++;
		}

		if (!connected)
		{
			ROS_ERROR("Lost connection with the robot. Please ensure the connecting cable is not loose. Retry calling shutdown_service after ensuring cable connections are proper.");
			return false;
		}

		if (state_ == RobotState::eMotion)
		{
			ROS_ERROR("Failed to shutdown the robot - still enabled");
			return false;
		}

		if (state_ == RobotState::eError)
		{
			ROS_ERROR("Failed to shutdown the robot - robot gone into error");
			return false;
		}

		ROS_INFO("Stopped the robot successfully.Disconnecting....");

		// Disconnecting of server
		mut_.lock();
		handler_->disconnect();
		mut_.unlock();

		std::this_thread::sleep_for(std::chrono::milliseconds(10));

		if (handler_->isConnected())
		{
			return false;
		}

		ROS_INFO("Disconnected successfully");

		return true;
	}

	/**
	 * @brief shutdown service - shutdown the robot
	 *
	 * @param req
	 * @param res
	 * @return true
	 * @return false
	 */
	bool RosWrapper::shutdownRobotCB_(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res)
	{
		if (req.data)
		{
			bool stop_status = shutdown();

			if (stop_status)
			{
				res.message = "Robot has been stopped and disabled";
				res.success = true;
				should_shutdown_ = true;
			}
			else
			{
				res.message = "Failed to stop robot. Re-attempt shutdown";
				res.success = false;
			}
		}
		else
		{
			ROS_INFO("Robot is already enabled.");
		}

		return true;
	}

	/**
	 * @brief update API fom apram server
	 *
	 * @param mode
	 * @return true
	 * @return false
	 */
	bool RosWrapper::updateAPI_(const std::string &mode)
	{
		if (mode == "velocity")
		{
			api_ = API::eExternalVelocityAPI;
		}
		else if (mode == "effort")
		{
			api_ = API::eExternalTorqueAPI;
		}
		else if (mode == "twist")
		{
			api_ = API::eJogAxisAPI;
		}
		else if (mode == "joint_trajectory")
		{
			api_ = API::eLinearVelocityAPI;
		}
		else
		{
			ROS_ERROR("Selected ros control mode is not valid");
			return false;
		}

		cconf_.controller = static_cast<int>(api_);
		return true;
	}

	/**
	 * @brief
	 *
	 * @return true
	 * @return false
	 */
	bool RosWrapper::shouldResetController_()
	{
		if (api_ == API::eJogAxisAPI || api_ == API::eExternalTorqueAPI)
		{
			return true;
		}

		return false;
	}

	/**
	 * @brief reset controller to jog axis
	 *
	 * @return true
	 * @return false
	 */
	bool RosWrapper::resetController_()
	{
		int count = 2;
		EventConfig event;
		event.event = static_cast<int>(Event::eResetConfig);

		hw_->getControllerConfig(cconf_);

		mut_.lock();
		handler_->appendConfig(cconf_);
		handler_->setUserConfig();
		handler_->setEventConfig(event);
		mut_.unlock();

		do
		{
			mut_.lock();
			handler_->writeData();
			count++;
			handler_->readData();
			mut_.unlock();
		} while (count < 2);

		std::this_thread::sleep_for(std::chrono::milliseconds(10));

		mut_.lock();
		handler_->clearData(UIDataType::eEventData);
		handler_->clearData(UIDataType::eConfigData);
		handler_->writeData();
		mut_.unlock();
		return true;
	}

	void RosWrapper::getSafetyMode_()
	{
		controller_safety_mode_map_ =
			{
				{API::eExternalVelocityAPI, 0},
				{API::eExternalTorqueAPI, 0},
				{API::eJogAxisAPI, 5},
				{API::eLinearVelocityAPI, 0}};
	}
}