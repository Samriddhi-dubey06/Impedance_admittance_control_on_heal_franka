#pragma once

#include <Eigen/Dense>
#include <cmath>


class FrictionModel
{
    public:
        FrictionModel() :
            coloumb_friction_(Eigen::Matrix<double, 6, 6>::Identity()),
            viscous_friction_(Eigen::Matrix<double, 6, 6>::Identity()),
            static_friction_(Eigen::Matrix<double, 6, 6>::Identity()),
            damping_kd_(Eigen::Matrix<double, 6, 6>::Identity()),
            velocity_threshold_(Eigen::Matrix<double, 6, 1>::Zero()),
            stribeck_velocity_(Eigen::Matrix<double, 6, 1>::Zero()),
            exponential_(Eigen::Matrix<double, 6, 1>::Zero())
        {
            
        }
        // The existing constructor that demands a namespace
        FrictionModel(const std::string &robot_namespace)
        {
            setConstantsForRobot(robot_namespace);
        }

        // Set constants based on the robot namespace
        void setConstantsForRobot(const std::string& robot_namespace)
        // {
        //     if (robot_namespace == "robotA")
        //     {
        //         coloumb_friction_( 0, 0 ) = 15;
        //         coloumb_friction_( 1, 1 ) = 15;
        //         coloumb_friction_( 2, 2 ) = 15;
        //         coloumb_friction_( 3, 3 ) = 15;
        //         coloumb_friction_( 4, 4 ) = 15;
        //         coloumb_friction_( 5, 5 ) = 0;

        //         viscous_friction_( 0, 0 ) = 5;
        //         viscous_friction_( 1, 1 ) = 20;
        //         viscous_friction_( 2, 2 ) = 20;
        //         viscous_friction_( 3, 3 ) = 30;
        //         viscous_friction_( 4, 4 ) = 30;
        //         viscous_friction_( 5, 5 ) = 0;

        //         static_friction_( 0, 0 ) = 15;
        //         static_friction_( 1, 1 ) = 15;
        //         static_friction_( 2, 2 ) = 15;
        //         static_friction_( 3, 3 ) = 10;
        //         static_friction_( 4, 4 ) = 15;
        //         static_friction_( 5, 5 ) = 0;

        //         stribeck_velocity_( 0 ) = 0.1;
        //         stribeck_velocity_( 1 ) = 0.1;
        //         stribeck_velocity_( 2 ) = 0.1;
        //         stribeck_velocity_( 3 ) = 0.1;
        //         stribeck_velocity_( 4 ) = 0.1;
        //         stribeck_velocity_( 5 ) = 0.1;

        //         velocity_threshold_( 0 ) = 0.001;
        //         velocity_threshold_( 1 ) = 0.001;
        //         velocity_threshold_( 2 ) = 0.001;
        //         velocity_threshold_( 3 ) = 0.001;
        //         velocity_threshold_( 4 ) = 0.001;
        //         velocity_threshold_( 5 ) = 0;

        //         damping_kd_( 0, 0 ) = 2;
        //         damping_kd_( 1, 1 ) = 2;
        //         damping_kd_( 2, 2 ) = 2;
        //         damping_kd_( 3, 3 ) = 2;
        //         damping_kd_( 4, 4 ) = 2;
        //         damping_kd_( 5, 5 ) = 0;

        //     }
        //  frist improvement
        {
            if (robot_namespace == "robotA")
            {
                coloumb_friction_( 0, 0 ) = 12;
                coloumb_friction_( 1, 1 ) = 5;
                coloumb_friction_( 2, 2 ) = 12;
                coloumb_friction_( 3, 3 ) = 12;
                coloumb_friction_( 4, 4 ) = 12;
                coloumb_friction_( 5, 5 ) = 0;

                viscous_friction_( 0, 0 ) = 17;
                viscous_friction_( 1, 1 ) = 8;
                viscous_friction_( 2, 2 ) = 17;
                viscous_friction_( 3, 3 ) = 17;
                viscous_friction_( 4, 4 ) = 15;
                viscous_friction_( 5, 5 ) = 0;

                static_friction_( 0, 0 ) = 0.1;
                static_friction_( 1, 1 ) = 0;
                static_friction_( 2, 2 ) = 0;
                static_friction_( 3, 3 ) = 0.1;
                static_friction_( 4, 4 ) = 0.1;
                static_friction_( 5, 5 ) = 0;

                stribeck_velocity_( 0 ) = 0.1;
                stribeck_velocity_( 1 ) = 0.1;
                stribeck_velocity_( 2 ) = 0.1;
                stribeck_velocity_( 3 ) = 0.1;
                stribeck_velocity_( 4 ) = 0.1;
                stribeck_velocity_( 5 ) = 0.1;

                velocity_threshold_( 0 ) = 0.001;
                velocity_threshold_( 1 ) = 0.001;
                velocity_threshold_( 2 ) = 0.001;
                velocity_threshold_( 3 ) = 0.001;
                velocity_threshold_( 4 ) = 0.001;
                velocity_threshold_( 5 ) = 0;

                damping_kd_( 0, 0 ) = 1;
                damping_kd_( 1, 1 ) = 1;
                damping_kd_( 2, 2 ) = 1;
                damping_kd_( 3, 3 ) = 1;
                damping_kd_( 4, 4 ) = 2;
                damping_kd_( 5, 5 ) = 0;

            }

            // {
            //     if (robot_namespace == "")
            //     {
            //         coloumb_friction_( 0, 0 ) = 12;
            //         coloumb_friction_( 1, 1 ) = 5;
            //         coloumb_friction_( 2, 2 ) = 12;
            //         coloumb_friction_( 3, 3 ) = 12;
            //         coloumb_friction_( 4, 4 ) = 12;
            //         coloumb_friction_( 5, 5 ) = 0;
    
            //         viscous_friction_( 0, 0 ) = 17;
            //         viscous_friction_( 1, 1 ) = 8;
            //         viscous_friction_( 2, 2 ) = 17;
            //         viscous_friction_( 3, 3 ) = 17;
            //         viscous_friction_( 4, 4 ) = 15;
            //         viscous_friction_( 5, 5 ) = 0;
    
            //         static_friction_( 0, 0 ) = 0.1;
            //         static_friction_( 1, 1 ) = 0;
            //         static_friction_( 2, 2 ) = 0;
            //         static_friction_( 3, 3 ) = 0.1;
            //         static_friction_( 4, 4 ) = 0.1;
            //         static_friction_( 5, 5 ) = 0;
    
            //         stribeck_velocity_( 0 ) = 0.1;
            //         stribeck_velocity_( 1 ) = 0.1;
            //         stribeck_velocity_( 2 ) = 0.1;
            //         stribeck_velocity_( 3 ) = 0.1;
            //         stribeck_velocity_( 4 ) = 0.1;
            //         stribeck_velocity_( 5 ) = 0.1;
    
            //         velocity_threshold_( 0 ) = 0.001;
            //         velocity_threshold_( 1 ) = 0.001;
            //         velocity_threshold_( 2 ) = 0.001;
            //         velocity_threshold_( 3 ) = 0.001;
            //         velocity_threshold_( 4 ) = 0.001;
            //         velocity_threshold_( 5 ) = 0;
    
            //         damping_kd_( 0, 0 ) = 1;
            //         damping_kd_( 1, 1 ) = 1;
            //         damping_kd_( 2, 2 ) = 1;
            //         damping_kd_( 3, 3 ) = 1;
            //         damping_kd_( 4, 4 ) = 2;
            //         damping_kd_( 5, 5 ) = 0;
    
            //     }
        
        // First Improvment
        // {
        //     if (robot_namespace == "robotA")
        //     {
        //         coloumb_friction_( 0, 0 ) = 12;
        //         coloumb_friction_( 1, 1 ) = 10;
        //         coloumb_friction_( 2, 2 ) = 9;
        //         coloumb_friction_( 3, 3 ) = 12;
        //         coloumb_friction_( 4, 4 ) = 10;
        //         coloumb_friction_( 5, 5 ) = 0;

        //         viscous_friction_( 0, 0 ) = 17;
        //         viscous_friction_( 1, 1 ) = 15;
        //         viscous_friction_( 2, 2 ) = 13;
        //         viscous_friction_( 3, 3 ) = 17;
        //         viscous_friction_( 4, 4 ) = 15;
        //         viscous_friction_( 5, 5 ) = 0;

        //         static_friction_( 0, 0 ) = 12;
        //         static_friction_( 1, 1 ) = 10;
        //         static_friction_( 2, 2 ) = 9;
        //         static_friction_( 3, 3 ) = 12;
        //         static_friction_( 4, 4 ) = 10;
        //         static_friction_( 5, 5 ) = 0;

        //         stribeck_velocity_( 0 ) = 0.1;
        //         stribeck_velocity_( 1 ) = 0.1;
        //         stribeck_velocity_( 2 ) = 0.1;
        //         stribeck_velocity_( 3 ) = 0.1;
        //         stribeck_velocity_( 4 ) = 0.1;
        //         stribeck_velocity_( 5 ) = 0.1;

        //         velocity_threshold_( 0 ) = 0.001;
        //         velocity_threshold_( 1 ) = 0.001;
        //         velocity_threshold_( 2 ) = 0.001;
        //         velocity_threshold_( 3 ) = 0.001;
        //         velocity_threshold_( 4 ) = 0.001;
        //         velocity_threshold_( 5 ) = 0;

        //         damping_kd_( 0, 0 ) = 30;
        //         damping_kd_( 1, 1 ) = 50;
        //         damping_kd_( 2, 2 ) = 100;
        //         damping_kd_( 3, 3 ) = 10;
        //         damping_kd_( 4, 4 ) = 100;
        //         damping_kd_( 5, 5 ) = 0;

        // }
        
        
        //     else if (robot_namespace == "robotB")
        //     {
        //         coloumb_friction_( 0, 0 ) = 30;
        //         coloumb_friction_( 1, 1 ) = 5;
        //         coloumb_friction_( 2, 2 ) = 15;
        //         coloumb_friction_( 3, 3 ) = 10;
        //         coloumb_friction_( 4, 4 ) = 10;
        //         coloumb_friction_( 5, 5 ) = 0;

        //         viscous_friction_( 0, 0 ) = 35;
        //         viscous_friction_( 1, 1 ) = 5;
        //         viscous_friction_( 2, 2 ) = 20;
        //         viscous_friction_( 3, 3 ) = 12;
        //         viscous_friction_( 4, 4 ) = 12;
        //         viscous_friction_( 5, 5 ) = 0;

        //         static_friction_( 0, 0 ) = 35;
        //         static_friction_( 1, 1 ) = 0;
        //         static_friction_( 2, 2 ) = 15;
        //         static_friction_( 3, 3 ) = 10;
        //         static_friction_( 4, 4 ) = 10;
        //         static_friction_( 5, 5 ) = 0;

        //         stribeck_velocity_( 0 ) = 0.1;
        //         stribeck_velocity_( 1 ) = 0.1;
        //         stribeck_velocity_( 2 ) = 0.1;
        //         stribeck_velocity_( 3 ) = 0.1;
        //         stribeck_velocity_( 4 ) = 0.1;
        //         stribeck_velocity_( 5 ) = 0.1;

        //         velocity_threshold_( 0 ) = 0.002;
        //         velocity_threshold_( 1 ) = 0.002;
        //         velocity_threshold_( 2 ) = 0.009;
        //         velocity_threshold_( 3 ) = 0.005;
        //         velocity_threshold_( 4 ) = 0.005;
        //         velocity_threshold_( 5 ) = 0;

        //         damping_kd_( 0, 0 ) = 1;
        //         damping_kd_( 1, 1 ) = 7;
        //         damping_kd_( 2, 2 ) = 1;
        //         damping_kd_( 3, 3 ) = 1;
        //         damping_kd_( 4, 4 ) = 1;
        //         damping_kd_( 5, 5 ) = 0;

        // }

        else
            {
                throw std::invalid_argument("Unknown robot namespace: " + robot_namespace);
            }
        }

        Eigen::Matrix<double, 6, 1> computeFrictionTorques(
            const Eigen::Matrix<double, 6, 1>& current_velocity);

    private:
        // Matrices for friction parameters
        Eigen::Matrix<double, 6, 6> coloumb_friction_;
        Eigen::Matrix<double, 6, 6> viscous_friction_;
        Eigen::Matrix<double, 6, 6> static_friction_;
        Eigen::Matrix<double, 6, 6> damping_kd_;

        // Vectors for thresholds and Stribeck
        Eigen::Matrix<double, 6, 1> velocity_threshold_;
        Eigen::Matrix<double, 6, 1> stribeck_velocity_;

        // Temporary storage for exponentials, if you want to store them 
        Eigen::Matrix<double, 6, 1> exponential_;
        Eigen::Matrix<double, 6, 1> sign;
        Eigen::Matrix<double, 6, 1> friction_torques;
};