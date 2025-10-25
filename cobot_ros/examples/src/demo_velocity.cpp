#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "demo_velocity");

    ros::NodeHandle n;

    ros::Publisher velocity_pub = n.advertise<std_msgs::Float64MultiArray>("/velocity_controller/command", 100);

    ros::Rate rate(1);

    ROS_INFO("starting demo_velocity");

    while (ros::ok())
    {
        std_msgs::Float64MultiArray velocity;

        velocity.data.push_back(0);
        velocity.data.push_back(0);
        velocity.data.push_back(0);
        velocity.data.push_back(0);
        velocity.data.push_back(0);
        velocity.data.push_back(0);

        velocity_pub.publish(velocity);

        ros::spinOnce();

        rate.sleep();
    }

    ROS_INFO("shutdown demo_velocity");

    return 0;
}