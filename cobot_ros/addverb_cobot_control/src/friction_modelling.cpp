#include "friction_modelling.h"
#include <cmath>

Eigen::Matrix<double, 6, 1> FrictionModel::computeFrictionTorques(
    const Eigen::Matrix<double, 6, 1>& current_velocity)
{
    // Determine sign based on velocity threshold
    for (int i = 0; i < 6; ++i)
    {
        if (current_velocity(i) > velocity_threshold_(i))
        {
            sign(i) =  1.0;
        }
        else if (current_velocity(i) < -velocity_threshold_(i))
        {
            sign(i) = -1.0;
        }
        else
        {
            sign(i) =  0.0;
        }

        // calculate the exponential (Stribeck) term
        exponential_(i) = std::exp(-std::abs(current_velocity(i)) 
                                   / stribeck_velocity_(i)) 
                          * sign(i);
    }

    // Combine Coulomb, viscous, and Stribeck+static friction
    friction_torques =
          ( coloumb_friction_ * sign )
        + ( viscous_friction_ * current_velocity )
        + ( (static_friction_ - coloumb_friction_) * exponential_ );

    return friction_torques;
}