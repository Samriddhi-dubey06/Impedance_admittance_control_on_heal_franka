Impedance–Admittance Control on Heal Franka

This repository contains the implementation of impedance and admittance control strategies on the Heal dual-arm robotic platform equipped with Franka Emika Panda manipulators. The work focuses on achieving compliant and adaptive interaction behaviors for physical human–robot and robot–environment interactions. This project is part of the research submitted to the  (ICRA) .

Overview

The goal of this work is to develop a unified control framework that combines impedance and admittance control for safe and stable dual-arm manipulation using the Heal bimanual robot. The implementation enables both arms to perform coordinated manipulation while maintaining compliance during contact with the environment.

Key Contributions
- Implementation of impedance control on the Franka arms for compliant trajectory tracking.
- Integration of admittance control to modify desired motion in response to external contact forces.
- Development of a hybrid control strategy allowing smooth transition between impedance and admittance modes.
- Experimental validation on the Heal bimanual robotic platform.
- Data acquisition and analysis of force, position, and velocity responses under different interaction conditions.

System Description

- Platform: Heal dual-arm robotic setup using two Franka Emika Panda manipulators.
- Sensors: Force/torque sensors at the end-effectors for contact force estimation.
- Control Interface: Implemented in Python/C++ using the Franka ROS interface and real-time control modules.
- The controllers are tested both in simulation and on the real hardware setup.

Implementation Details

1. Impedance Control: 
   The controller regulates the end-effector motion based on a desired stiffness–damping relationship, allowing the arm to react compliantly to external forces while maintaining trajectory accuracy.

2. Admittance Control:
   The controller modifies the reference trajectory in response to sensed forces, effectively changing the robot's motion behavior based on environmental interactions.

3. Hybrid Mode:
   A hybrid switching mechanism is developed to combine impedance and admittance behavior, enabling robust contact handling and adaptive motion during physical interaction.

Results and Validation

- The proposed control framework allows smooth force-regulated manipulation tasks with stable contact behavior.
- Experimental results show compliant responses under varying stiffness and damping parameters.
- The implementation has been demonstrated on the Heal bimanual robot for object interaction and cooperative manipulation tasks.

Repository Structure

controllers/
    impedance_controller.py
    admittance_controller.py
    hybrid_controller.py
scripts/
    run_impedance_control.py
    run_admittance_control.py
    run_hybrid_mode.py
config/
    control_gains.yaml
    trajectories/
README.md

How to Run

1. Clone the repository:
   git clone https://github.com/Samriddhi-dubey06/Impedance_admittance_control_on_heal_franka
   cd Impedance_admittance_control_on_heal_franka

2. Install the required dependencies:
   pip install numpy rospy matplotlib scipy

3. Launch the control node:
   roslaunch heal_control run_hybrid.launch

4. Modify stiffness and damping parameters in config/control_gains.yaml to observe different compliance behaviors.

Author

Samriddhi Dubey
B.Tech, Mechanical Engineering, IIT Gandhinagar
Research Areas: Robotics, Impedance/Admittance Control, Dual-Arm Manipulation, Human–Robot Interaction
