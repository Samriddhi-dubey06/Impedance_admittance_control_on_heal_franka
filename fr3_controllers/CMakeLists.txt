cmake_minimum_required(VERSION 3.5)
project(fr3_controllers)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  controller_interface
  dynamic_reconfigure
  eigen_conversions
  franka_hw
  franka_gripper
  geometry_msgs
  hardware_interface
  joint_limits_interface
  tf
  tf_conversions
  message_generation
  pluginlib
  realtime_tools
  roscpp
  rospy
  urdf
  visualization_msgs
)

find_package(Eigen3 REQUIRED)
find_package(Franka 0.9.0 QUIET)
if(NOT Franka_FOUND)
  find_package(Franka 0.8.0 REQUIRED)
endif()

# merge Franka + INCLUDE_DIRS in topological order
list_insert_in_workspace_order(catkin_INCLUDE_DIRS ${Franka_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})

add_message_files(FILES
  JointTorqueComparison.msg
)

generate_messages()

generate_dynamic_reconfigure_options(
  cfg/compliance_param.cfg
  cfg/desired_mass_param.cfg
  cfg/dual_arm_compliance_param.cfg
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES fr3_controllers
  CATKIN_DEPENDS
    controller_interface
    dynamic_reconfigure
    eigen_conversions
    franka_hw
    franka_gripper
    geometry_msgs
    hardware_interface
    joint_limits_interface
    tf
    tf_conversions
    message_runtime
    pluginlib
    realtime_tools
    roscpp
    urdf
    visualization_msgs
  DEPENDS Franka
)

add_library(fr3_controllers
  src/eoi_controller.cpp
  src/cartesian_pose_controller.cpp
  src/joint_position_controller.cpp
  src/joint_velocity_controller.cpp
  src/variable_impedance_controller.cpp
  src/cartesian_impedance_controller.cpp
  src/joint_impedance_controller.cpp
  src/dob.cpp
)

catkin_install_python(PROGRAMS
  scripts/ee_pose.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_dependencies(fr3_controllers
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  ${PROJECT_NAME}_generate_messages_cpp
  ${PROJECT_NAME}_gencpp
  ${PROJECT_NAME}_gencfg
)

target_link_libraries(fr3_controllers PUBLIC
  ${Franka_LIBRARIES}
  ${catkin_LIBRARIES}
)

target_include_directories(fr3_controllers SYSTEM PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
target_include_directories(fr3_controllers PUBLIC
  include
)

## Installation
install(TARGETS fr3_controllers
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES fr3_controllers.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

## Tools
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/ClangTools.cmake OPTIONAL
  RESULT_VARIABLE CLANG_TOOLS
)
if(CLANG_TOOLS)
  file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
  file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
  )
  add_format_target(fr3_controllers FILES ${SOURCES} ${HEADERS})
  add_tidy_target(fr3_controllers
    FILES ${SOURCES}
    DEPENDS fr3_controllers
  )
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/PepTools.cmake OPTIONAL
  RESULT_VARIABLE PEP_TOOLS
)
if(PEP_TOOLS)
  file(GLOB_RECURSE PYSOURCES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.py)
  add_pyformat_target(franka_control FILES ${PYSOURCES})
endif()
