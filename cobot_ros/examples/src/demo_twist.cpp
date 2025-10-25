#include "ros/ros.h"
#include "geometry_msgs/Twist.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "demo_twist");

    ros::NodeHandle n;

    ros::Publisher twist_pub = n.advertise<geometry_msgs::Twist>("/twist_controller/command", 100);

    ros::Rate rate(10000000);

    ROS_INFO("starting demo_twist");

    while (ros::ok())
    {
        geometry_msgs::Twist twist;

        twist.linear.x = 0;
        twist.linear.y = 0;
        twist.linear.z = 0;
        twist.angular.x = 0;
        twist.angular.y = 0;
        twist.angular.z = 0;

        twist_pub.publish(twist);

        ros::spinOnce();

        rate.sleep();
    }

    ROS_INFO("shutdown demo_twist");

    return 0;
}