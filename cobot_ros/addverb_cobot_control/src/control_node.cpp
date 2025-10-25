/**
 * @file ros_wrapper.cpp
 * @author Gonna Yaswanth (yaswanth.gonna@addverb.com)
 * @brief Cobot ROS wrapper
 * @version 0.1
 * @date 2024-09-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "ros_wrapper.h"

// ROS header files
#include <ros/ros.h>

// Data handler
#include <data_handler.h>

int main(int argc, char **argv)
{
    // initialise ROS node
    ros::init(argc, argv, "control_node");

    // node handler
    ros::NodeHandle nh;

    // data handler
    std::shared_ptr<DataHandler> data_handler = std::make_shared<DataHandler>();

    if (!data_handler->setup())
    {
        ROS_ERROR("Failed to setup connection with robot");
        return 1;
    }
    // connect with Robot
    int connect_attempt = 0;

    while (connect_attempt < 60)
    {
        if (data_handler->connect())
        {
            break;
        }

        ROS_WARN("Attempt to establish connection with robot FAILED. Please ensure the cable is connected and is not loose.");

        connect_attempt++;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    if (!data_handler->isConnected())
    {
        ROS_ERROR("Failed to connect to robot. Recheck if the robot is powered on, and the wires are connected");
        return 1;
    }

    ROS_INFO("Connected with the robot");

    ros::AsyncSpinner spinner(3);
    spinner.start();

    addverb::RosWrapper interface(nh, data_handler);

    // Waiting for ROS shutdown
    ros::waitForShutdown();

    return 0;
}