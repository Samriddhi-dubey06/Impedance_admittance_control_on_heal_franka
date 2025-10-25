'''
This code publishes the force bimanually on the object to maintatin the force closure 

'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import sys
sys.path.insert(0, '/home/iitgn-robotics/ds_yash/bimanual_ws/src')

# Import the two controllers from utils
from dynamic_box_lifting.utils.heal_controller_V1 import HealController
from dynamic_box_lifting.utils.franka_controller_V1 import FrankaController


def main():
    rospy.init_node("force_velocity_dual_robot")

    # You can override defaults via kwargs if needed:
    heal = HealController(
        # example override:
        # topic_ft="/ft_sensor",
        lambda_star=np.array([-7.0, 0, 0, 0, 0, 0], dtype=float)
    )
    franka = FrankaController(
        # example override:
        lambda_star=np.array([0, -7.0, 0, 0, 0, 0], dtype=float)
    )

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        heal.step()
        franka.step()
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
