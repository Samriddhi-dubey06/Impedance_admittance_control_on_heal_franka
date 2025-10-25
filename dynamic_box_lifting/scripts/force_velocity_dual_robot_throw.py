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
from dynamic_box_lifting.utils.heal_controller_throw import HealController
from dynamic_box_lifting.utils.franka_controller_throw import FrankaController
from dynamic_box_lifting.utils.throw_orchestrator import ThrowOrchestrator


def main():
    rospy.init_node("force_velocity_dual_robot")

    heal = HealController(lambda_star=np.array([-7.0, 0, 0, 0, 0, 0], dtype=float))
    franka = FrankaController(lambda_star=np.array([0, -7.0, 0, 0, 0, 0], dtype=float))

    orchestrator = ThrowOrchestrator()  # exposes /force_velocity_dual_robot/throw_now

    rate = rospy.Rate(200)  # match orchestrator pub rate
    while not rospy.is_shutdown():
        heal.step()
        franka.step()
        orchestrator.step()
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass