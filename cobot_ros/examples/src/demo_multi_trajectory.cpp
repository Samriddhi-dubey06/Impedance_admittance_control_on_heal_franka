#include "ros/ros.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "trajectory_msgs/JointTrajectoryPoint.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "demo_multi_trajectory");

    ros::NodeHandle n;

    ros::Publisher joint_trajectory_pub = n.advertise<trajectory_msgs::JointTrajectory>("/joint_trajectory_controller/command", 100);

    ros::Rate rate(1);

    ROS_INFO("starting demo_multi_trajectory");

    int counter = 0;

    trajectory_msgs::JointTrajectory trajectory;

    trajectory_msgs::JointTrajectoryPoint point;

    point.positions.push_back(0);
    point.positions.push_back(0);
    point.positions.push_back(0);
    point.positions.push_back(0);
    point.positions.push_back(0);
    point.positions.push_back(0);
    point.time_from_start = ros::Duration(7);
    trajectory.points.push_back(point);
    point.time_from_start = ros::Duration(14);
    trajectory.points.push_back(point);
    point.time_from_start = ros::Duration(21);
    trajectory.points.push_back(point);

    while (ros::ok())
    {
        if (counter % 10 == 0)
        {
            joint_trajectory_pub.publish(trajectory);
            std::cout << "published\n";
        }

        ros::spinOnce();
        rate.sleep();
        counter++;
    }

    ROS_INFO("shutdown demo_multi_trajectory");

    return 0;
}