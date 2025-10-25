cmake_minimum_required(VERSION 3.5)
project(ds_control)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package()

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_python_setup()

catkin_install_python(PROGRAMS  
    scripts/dual_ds.py
    scripts/main.py
    scripts/testV1.py
    scripts/testV2.py
    scripts/testV3.py
    scripts/testV4.py
    scripts/testV5.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

include_directories(
  ${catkin_INCLUDE_DIRS}
)