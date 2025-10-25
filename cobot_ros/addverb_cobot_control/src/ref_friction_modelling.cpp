#include <iostream>
#include "system_manager.h"
#include "kd_impl.h"
#include <iomanip>
Eigen::Matrix<double, 6, 6>coloumb_friction = Eigen::Matrix<double, 6, 6>::Identity( );
Eigen::Matrix<double, 6, 6>viscous_friction = Eigen::Matrix<double, 6, 6>::Identity( );
Eigen::Matrix<double, 6, 6>static_friction = Eigen::Matrix<double, 6, 6>::Identity( );
Eigen::Matrix<double, 6, 6>damping_kd = Eigen::Matrix<double, 6, 6>::Identity( );
Eigen::Matrix<double, 6, 1>exponential;
Eigen::Matrix<double, 6, 1>velocity_threshold;
Eigen::Matrix<double, 6, 1>stribeck_velocity;
Eigen::Matrix<double, 6, 1> cur_pos, prev_pos;


void setConstants( ) {
    coloumb_friction( 0, 0 ) = 35;
    coloumb_friction( 1, 1 ) = 10;
    coloumb_friction( 2, 2 ) = 15;
    coloumb_friction( 3, 3 ) = 10;
    coloumb_friction( 4, 4 ) = 10;
    coloumb_friction( 5, 5 ) = 0;

    viscous_friction( 0, 0 ) = 50;
    viscous_friction( 1, 1 ) = 10;
    viscous_friction( 2, 2 ) = 25;
    viscous_friction( 3, 3 ) = 20;
    viscous_friction( 4, 4 ) = 20;
    viscous_friction( 5, 5 ) = 0;

    static_friction( 0, 0 ) = 43;
    static_friction( 1, 1 ) = 15;
    static_friction( 2, 2 ) = 20;
    static_friction( 3, 3 ) = 10;
    static_friction( 4, 4 ) = 10;
    static_friction( 5, 5 ) = 0;

    stribeck_velocity( 0 ) = 0.1;
    stribeck_velocity( 1 ) = 0.1;
    stribeck_velocity( 2 ) = 0.1;
    stribeck_velocity( 3 ) = 0.1;
    stribeck_velocity( 4 ) = 0.1;
    stribeck_velocity( 5 ) = 0.1;

    velocity_threshold( 0 ) = 0.002;
    velocity_threshold( 1 ) = 0.002;
    velocity_threshold( 2 ) = 0.009;
    velocity_threshold( 3 ) = 0.002;
    velocity_threshold( 4 ) = 0.002;
    velocity_threshold( 5 ) = 0;

    damping_kd( 0, 0 ) = 1;
    damping_kd( 1, 1 ) = 3;
    damping_kd( 2, 2 ) = 1;
    damping_kd( 3, 3 ) = 1;
    damping_kd( 4, 4 ) = 1;
    damping_kd( 5, 5 ) = 0;


}

Eigen::Matrix<double, 6, 1> fullFrictionModel( Eigen::Matrix<double, 6, 1>& current_velocity, Eigen::Matrix<double, 6, 1>& velocity_threshold ) {

    Eigen::Matrix<double, 6, 1> sign;
    Eigen::Matrix<double, 6, 1> friction_torques;
    std::cout << std::setprecision( 5 );
    std::cout << current_velocity( 0 ) << "\n";
    // set the signs for all joints
    for ( int i = 0; i < 6; i++ ) {
        if ( current_velocity( i ) > velocity_threshold( i ) ) {
            sign( i ) = 1;
        }
        else if ( current_velocity( i ) < -velocity_threshold( i ) ) {
            sign( i ) = -1;
        }
        else {
            sign( i ) = 0;
        }

        // calculate the exponential terms for all joints
        exponential( i ) = exp( -std::abs( current_velocity( i ) ) / stribeck_velocity( i ) ) * sign( i );

    }

    // find the overall friction torques for all joints
    friction_torques = ( coloumb_friction * sign ) + ( viscous_friction * current_velocity ) + ( ( static_friction - coloumb_friction ) * exponential );

    return friction_torques;
}



int main( ) {

    SystemManager sys;
    Config config;
    AlliedData interrupt;
    RobotData rd;
    KdImpl kd;
    kd.setup( );
    config.reset_controller = static_cast< int >( ControllerEnum::eExternalTorque );
    config.safety_type = 1;

    if ( !sys.setupRobot( config ) ) {
        std::cout << "Failed to setup the robot\n";
        return 0;
    }

    double target_vel = 0.5;
    double vel = 0;

    int joint_id = 4;

    Eigen::Matrix<double, 6, 1> jpos, jvel, jtor;

    setConstants( );

    while ( true ) {
        sys.getRobotData( rd );
        // std::cout << rd.jtor [joint_id] << " " << rd.jvel [joint_id] << "\n";
        for ( size_t i = 0; i < 6; i++ )
        {
            jpos( i ) = rd.jpos [i];
            jvel( i ) = rd.jvel [i];
        }
        cur_pos = jpos;

        // std::cout << jpos.transpose( ) << "\n";
        kd.setVariables( jpos );
        kd.doDyn( jvel, jvel * 0 );
        kd.getDynVec( jtor );

        // jtor( joint_id ) = jtor( joint_id ) + frictionModel( jvel( joint_id ), velocity_threshold );
        // std::cout << ( jtor( joint_id ) + frictionModel( jvel( joint_id ), velocity_threshold ) - damping_kd * jvel( joint_id ) ) << "\n";

        Eigen::Matrix<double, 6, 1> friction_torques = fullFrictionModel( jvel, velocity_threshold );
        // std::cout << ( jtor + friction_torques - damping_kd * jvel ).transpose( ) << "\n";

        Eigen::Matrix<double, 6, 1>command_torques = jtor + friction_torques - ( damping_kd * jvel );
        for ( size_t i = 0; i < 6; i++ )
        {

            interrupt.external_control_data [i] = command_torques( i );
 }


        // if ( vel < target_vel ) {
        //     vel += 0.001;
        // }
        // interrupt.external_control_data = { 0,0,0,-vel,0,0 };


        if ( !sys.doControl( interrupt ) ) {
            std::cout << "failed to control the robot\n";
            return 0;
        }
        prev_pos = cur_pos;
        // usleep( 1000);
    }


    return 0;
}


