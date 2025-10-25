#include "ros/ros.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "trajectory_msgs/JointTrajectoryPoint.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "demo_joint_trajectory");

    ros::NodeHandle n;

    ros::Publisher joint_trajectory_pub = n.advertise<trajectory_msgs::JointTrajectory>("/joint_trajectory_controller/command", 100);

    ros::Rate rate(1);

    ROS_INFO("starting demo_joint_trajectory");

    int counter = 0;

    while (ros::ok())
    {
        trajectory_msgs::JointTrajectory trajectory;
        trajectory_msgs::JointTrajectoryPoint point;

        switch (counter)
        {
        case 0:
        {
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.time_from_start = ros::Duration(7);
            trajectory.points.push_back(point);
            joint_trajectory_pub.publish(trajectory);

            break;
        }
        case 1:
        {
            point.positions.push_back(-0.41);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.time_from_start = ros::Duration(7);
            trajectory.points.push_back(point);
            joint_trajectory_pub.publish(trajectory);

            break;
        }
        case 2:
        {
            point.positions.push_back(-0.41);
            point.positions.push_back(0.23);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.time_from_start = ros::Duration(7);
            trajectory.points.push_back(point);
            joint_trajectory_pub.publish(trajectory);

            break;
        }
        case 3:
        {
            point.positions.push_back(-0.11);
            point.positions.push_back(0.23);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.time_from_start = ros::Duration(7);
            trajectory.points.push_back(point);
            joint_trajectory_pub.publish(trajectory);

            break;
        }
        case 4:
        {
            point.positions.push_back(-0.11);
            point.positions.push_back(0.11);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.time_from_start = ros::Duration(7);
            trajectory.points.push_back(point);
            joint_trajectory_pub.publish(trajectory);

            break;
        }
        case 5:
        {
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.positions.push_back(0);
            point.time_from_start = ros::Duration(7);
            trajectory.points.push_back(point);
            joint_trajectory_pub.publish(trajectory);

            break;
        }
        default:
            break;
        }

        ros::spinOnce();
        rate.sleep();
        counter++;
    }

    ROS_INFO("shutdown demo_joint_trajectory");

    return 0;
}