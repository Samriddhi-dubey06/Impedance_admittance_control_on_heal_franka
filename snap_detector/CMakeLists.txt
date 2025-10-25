cmake_minimum_required(VERSION 3.5)
project(snap_detector)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
)

catkin_package()

catkin_python_setup()

catkin_install_python(PROGRAMS scripts/snap_detector_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})