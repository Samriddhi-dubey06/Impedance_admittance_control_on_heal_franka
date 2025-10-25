# cobot_ros


## Dependencies

1. Ubuntu 20.04 LTS
2. ROS Noetic (if not installed already, please install using the instructions provided [here](http://wiki.ros.org/noetic/Installation/Ubuntu))
3. Install ROS dependencies using the following command:
    ```
    sudo apt-get install liborocos-kdl-dev ros-noetic-ros-control ros-noetic-joint-state-controller ros-noetic-effort-controllers ros-noetic-position-controllers ros-noetic-velocity-controllers
    ```
4. Install comm_dependencies 
    ```
    mkdir ~/ros_installer
    cd ~/ros_installer
    ```

    Copy the downloaded zip file named comm_dependencies.zip and dependency_installer in this folder named ros_installer
    To unzip these dependencies in the folder, run the following commands :

    ```
    sudo apt-get install unzip
    unzip comm_dependencies.zip
    rm -r comm_dependencies.zip
    ```
    To install these dependencies in you system, run the following commands :

    ```
    sudo chmod a+rwx dependency_installer
    ./dependency_installer
    ```

    Once the dependencies are installed, you can delete this folder. To do that, run the following commands :
    ```
    cd ~
    rm -rf ros_installer
    ```

## Installation and Compilation of the ROS Package

1. Create a ROS workspace
    ```
    mkdir -p ~/Heal_ws/src
    cd ~/Heal_ws/src
    ```
2. To setup the ROS Package, first copy the zip files named cobot_ros.zip and mesh.zip in the `src` folder and then run the following commands :
    ```
    unzip cobot_ros.zip
    unzip mesh.zip
    mv -i mesh cobot_ros/addverb_cobot_description/
    rm -r mesh.zip
    rm -r cobot_ros.zip
    ```
3. Build 
    ```
    cd ~/Heal_ws
    catkin_make
    source devel/setup.bash
    ```


<a name="setup-robot"></a>
## Setup Robot

1. Make your IP static and set it to the following
    ```
    Address: 192.168.1.21
    Netmask: 255.255.255.0
    ```
2. Connect the LAN cable to the robot and then SSH into the robot
   
    In case you do not have ssh installed on your system, run the following commands to install it :
    ```
    sudo apt-get install openssh-client
    ```
    Once you have SSH Client installed, run :

    ``` 
    ssh cobot@192.168.1.25
    > password: cobot
    ```

3. Build the application layer on-board the robot :

   Copy the downloaded zip folder named application_layer.zip on the robot PC at the location `/home/cobot`. After that, build it by following these commands : 
    ```
    unzip -r application_layer.zip
    rm -rf application_layer.zip
    cd application_layer
    mkdir build 
    cd build
    cmake ..
    make 
    ```

## Running the robot using the ROS Package

1. Connect the LAN cable to the robot and then SSH into the robot
   
    ``` 
    ssh cobot@192.168.0.12
    > password: cobot
    ```

2. To run the robot through ROS package, follow these commands :
    ```
    cd tests/build
    sudo ./heal_server
    ```

## Configuration

The following robot parameters/control modes can be configured using the YAML files as mentioned.
_NOTE: The configuration step must be done before proceeding to [Execution](#execution) section._


### Payload
    
Payload defines the properties of the attachment at the end effector of the robot. To modify the payload parameters : 
```
nano ~/catkin_ws/src/cobot_ros/addverb_cobot_control/config/payload_param.yaml 
```

Description of parameters: 
- **payload_status** : Expects a boolean value (`true`/`false`). Use `true`, when you have an attachment at the end-effector, `false` otherwise. If this field is set to `false`, further parameters need not be set in the YAML. 
- **gripper_type** : Expects an integer between 1 to 4, indicating the make of the gripper.
- **mass** : Expects a `double` value, indicating the mass ($kg$) of the attachment.
- **comx, comy, comz** : Expect a `double` value, representing the Centre of Mass (m) of the cumulative system attached at the end-effector.
- **Ixx, Iyy, Izz, Ixy, Ixz, Iyz** : Expect a `double` value, representing the Moment of Inertia ($kg.m^2$) of the cumulative system attached at the end-effector.


<a name="control-mode"></a>
### Control Mode

Control Mode defines the mode in which you want to operate the robot. To modify the control mode : 
```
nano ~/catkin_ws/src/cobot_ros/addverb_cobot_control/config/ros_joint_controllers.yaml 
```

Description of parameters: 
- **ros_control_mode** : Expects a `string` indicating the mode of control (`velocity`/`effort`/`twist`/`joint_trajectory`).


<a name="execution"></a>
## Execution

Open two terminals (hereafter referred to as Terminal 1, and Terminal 2). Execute the following commands in each terminal :
```
cd ~/catkin_ws
source devel/setup.bash
```

1. Terminal 1: Launch the controllers by running the following command:
    ```
    roslaunch addverb_cobot_control bringup.launch
    ```
    _NOTE: `RViz` will launched along with `bringup.launch` by default. This can be disabled by specifying `gui:=false` as argument to the above command._
2. Terminal 2: Execute the *default control mode* (i.e., `velocity`). The velocity commands, that is, each element of the array - `data: [0, 0, 0, 0, 0, 0]` directly controls each joint of the robot. Please run the following node:
    ```
    rosrun examples demo_velocity
    ```
    By default all joint velocities are set to zero. Therefore you will not notice any motion. To modify the control commands you can edit the file at the path :
    ```
    nano ~/ros_ws/src/cobot_ros/examples/src/demo_velocity.cpp
    ```


## Controllers

  - Mode 1 (`velocity`): This is the *default controller*. The velocity commands, that is, each element of the array - `data: [0, 0, 0, 0, 0, 0]` directly controls each joint of the robot. Before proceeding further, please ensure that the correct controller is set or update as mentioned in [Control Mode](#control-mode) section. Later, run the following command:
    ```
    rosrun examples demo_velocity
    ```
    By default all joint velocities are set to zero. Therefore you will not notice any motion. To modify the control commands you can edit the file at the path :
    ```
    nano ~/ros_ws/src/cobot_ros/examples/src/demo_velocity.cpp
    ```
  - Mode 2 (`effort`): The effort commands, that is, each element of the array - `data: [0, 0, 0, 0, 0, 0]` directly controls each joint of the robot. Before proceeding further, please ensure that the correct controller is set or update as mentioned in [Control Mode](#control-mode) section. Later, run the following command to directly give effort commands via terminal:
  ```
  rostopic pub -1 /effort_controller/command  std_msgs/Float64MultiArray "data: [-0.6,-18.946, 18.946, 0.892178, 0.386, 1.29]"
  ```
  - Mode 3 (`twist`): Each element of the `Twist` message, controls the direction (`x`, `y`, `z` axes) along which or about which the robot's end-effector will move. Each element of the twist message can only have a value belonging to the set `{-1, 0, 1}`, where -1 means movement along negative direction of chosen axis, +1 means movement along positive direction of chosen axis and 0 means no movement. Before proceeding further, please ensure that the correct controller is set or update as mentioned in [Control Mode](#control-mode) section. Later, run the following command:
    ```
    rosrun examples demo_twist
    ```
    By default all the twist components are set to zero, to modify the control commands you can edit the file at the path :
    ```
    gedit ~/ros_ws/src/cobot_ros/examples/src/demo_twist.cpp
    ```
  - Mode 4 (`joint_trajectory`): 
    - Case I: It demonstrates sending goal postions one by one. Each `JointTrajectory` message, provides goal position along with the time desired to achieve the goal. Before proceeding further, please ensure that the correct controller is set or update as mentioned in [Control Mode](#control-mode) section. Later, run the following command:
        ```
        rosrun examples demo_joint_trajectory
        ```
        To modify the control commands you can edit the file at the path :
        ```
        nano ~/ros_ws/src/cobot_ros/examples/src/demo_joint_trajectory.cpp
        ```

    - Case II: It demonstrates sending goal postions all at once. Each `JointTrajectory` message, provides all goal positions along with the time desired to achieve each goal position. **NOTE** : A minimum of _three_ goal positions must be specified in this case. Also, the time desired to achieve the goal position must be incremental, i.e., Please ensure that time given for (i+1)th goal position must be greater than time specified for ith goal position. Before proceeding further, please ensure that the correct controller is set or update as mentioned in [Control Mode](#control-mode) section. Later, run the following command:
        ```
        rosrun examples demo_multi_trajectory
        ```
        To modify the control commands you can edit the file at the path :
        ```
        nano ~/ros_ws/src/cobot_ros/examples/src/demo_multi_trajectory.cpp
        ```


## Gripper Control

- Open grippper
    - Demo node:
        ```
        ros2 run examples demo_gripper_open
        ```
    - Terminal command
        ```
        ros2 action send_goal /GripperAction addverb_cobot_msgs/action/Gripper "{event: 2, grasp_force: 0}"
        ```
- Close grippper
    - Demo node:
        ```
        ros2 run examples demo_gripper_close
        ```
    - Terminal command
        ```
        ros2 action send_goal /GripperAction addverb_cobot_msgs/action/Gripper "{event: 3, grasp_force: 50}"
        ```
        To changes gripper grasping force, change values of `goal_msg.grasp_force` in `src/examples/demo_gripper_close.cpp line:43`.


## Troubleshooting

1. Go back to the base pose (if required): The following command should be executed inside the cobot. Therefore, please do `ssh cobot@192.168.0.12` as suggested at [Setup Robot](#setup-robot) section.
    ``` 
    sudo ./tests/base_rigid
    
    ```
    
## Setting up the effort controller for heal
1. Go to heal_ws/src/cobot_ros/addverb_cobot_control/config, inside default controllers.yaml set the ros_control_mode to "effort".
2. Go to heal_ws/src/cobot_ros/addverb_cobot_control/include, inside addverb_cobot_hw.h add necessary packages and libraries:

// ROS control

#include <std_msgs/Float64MultiArray.h>
#include <urdf/model.h>
#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/jntarray.hpp>

3. Go to heal_ws/src/cobot_ros/addverb_cobot_control/launch, inside bringup.launch in "args" set "effort_controller".

4. Go to heal_ws/src/cobot_ros/addverb_cobot_control/src, inside addverb_cobot_hw.cpp add gravity compensation part inside the effort_controller section, that basically incudes the mass, coriolis and gravity torques.
5. Go to heal_ws/src/cobot_ros/addverb_cobot_control/package.xml, set up the required dependencies like:

 <buildtool_depend>catkin</buildtool_depend>
  <build_depend>addverb_cobot_msgs</build_depend>
  <build_depend>roscpp</build_depend>
  <build_depend>actionlib</build_depend>
  <build_export_depend>roscpp</build_export_depend>
  <build_depend>sensor_msgs</build_depend>
  <build_depend>std_msgs</build_depend>
  <build_depend>joint_limits_interface</build_depend>
  <build_depend>addverb_cobot_controllers</build_depend>
  
  <exec_depend>addverb_cobot_msgs</exec_depend>
  <exec_depend>roscpp</exec_depend>
  <exec_depend>actionlib</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>joint_limits_interface</exec_depend>
  <exec_depend>addverb_cobot_controllers</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  
  
6. Make changes in heal_ws/src/cobot_ros/addverb_cobot_control/CMakeLists.txt as well.

7. # Steps to Check and Launch the Effort Controller

Follow these steps to ensure the `effort_controller` is working correctly. Once completed, you can use the `free_drive` mode to observe changes in torque values.

### Step 1: Verify connectivity by pinging the robot
```
ping 192.168.1.25
```

### Step 2: Establish an SSH connection to the robot
```
ssh cobot@192.168.1.25
```

### Step 3: Navigate to the test build directory
```
cd tests/build
```

### Step 4: Check the available files
```
ls
```

### Step 5: Run the Heal server with sudo permissions
```
sudo ./heal_server
```

### Step 6: Open another terminal and launch the effort_controller
```
cd Heal_ws

```

```
source devel/setup.bash
```

```
roslaunch addverb_cobot_control bringup.launch
```

After completing the steps, the `effort_controller` will be launched. While using `free_drive`, you can see the changes in torque values.




  
  
  






