cmake_minimum_required(VERSION 3.5)
project(addverb_cobot_control)

# Compile as C++17
add_compile_options(-std=c++17)

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  addverb_cobot_msgs
  roscpp
  actionlib
  sensor_msgs
  controller_manager
  hardware_interface
  control_msgs
  addverb_cobot_controllers
  joint_limits_interface
  kdl_parser
  urdf
)

# Find Orocos KDL
find_package(orocos_kdl REQUIRED)

# Backend configuration (Always use Addverb Backend)
SET(COBOT_UI_BACKEND_INCLUDE_DIRS /opt/addverb/cobot_backend/include)
SET(BACKEND_BUILD_DEPEND /opt/addverb/cobot_backend/include/include/struct_mapping)
SET(BACKEND_PATH /opt/addverb/cobot_backend)
message(STATUS "Using Addverb Backend: /opt/addverb/cobot_backend")


SET(COBOT_UI_BACKEND_LIBRARIES)
SET(DEPEND_LIB_LIST)
LIST(APPEND DEPEND_LIB_LIST 
socket_communication
socket_communication_json
socket_client
socket_server
socket_comm_wrapper_baseline
socket_comm_wrapper
socket_client_comm_wrapper
socket_server_comm_wrapper 
network_manager
data_handler 
backend_executor 
data_adapter
data_parser
listener
http_server
websocket_server
shared_state
data_store 
error_manager_impl)

foreach(LIB ${DEPEND_LIB_LIST})
    find_library(DEPEND_LIB_${LIB} ${LIB}
      PATHS ${BACKEND_PATH}
      NO_DEFAULT_PATH)
    if(NOT DEPEND_LIB_${LIB})
        message(FATAL_ERROR "Library ${LIB} not found in ${BACKEND_PATH}")
    endif()
    LIST(APPEND COBOT_UI_BACKEND_LIBRARIES ${DEPEND_LIB_${LIB}})
    MESSAGE("Lib: ${LIB}")
    MESSAGE("Found Lib: ${DEPEND_LIB_${LIB}}")
endforeach(LIB)

# catkin specific configuration
catkin_package(
  INCLUDE_DIRS include ${COBOT_UI_BACKEND_INCLUDE_DIRS}
  CATKIN_DEPENDS addverb_cobot_msgs sensor_msgs
)

# Build
include_directories(
  include
  ${BACKEND_BUILD_DEPEND}
  ${COBOT_UI_BACKEND_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${orocos_kdl_INCLUDE_DIRS}
  ${kdl_parser_INCLUDE_DIRS}
  ${urdf_INCLUDE_DIRS}
)

# Addverb cobot hardware interface
add_library(addverb_cobot_hw 
    src/addverb_cobot_hw.cpp 
    src/friction_modelling.cpp 
)
target_link_libraries(addverb_cobot_hw 
${COBOT_UI_BACKEND_LIBRARIES}
${catkin_LIBRARIES}
${orocos_kdl_LIBRARIES}
${kdl_parser_LIBRARIES}
${urdf_LIBRARIES}
)
target_include_directories(addverb_cobot_hw PUBLIC include/addverb_cobot_control)

# CONTROL NODE
add_executable(control_node src/control_node.cpp)
target_link_libraries(control_node
${catkin_LIBRARIES}
${COBOT_UI_BACKEND_LIBRARIES}
${orocos_kdl_LIBRARIES}
${kdl_parser_LIBRARIES}
ros_wrapper
)
target_include_directories(control_node PUBLIC include/addverb_cobot_control)

# ROS WRAPPER
add_library(ros_wrapper SHARED src/ros_wrapper.cpp)
target_link_libraries(ros_wrapper PUBLIC 
${COBOT_UI_BACKEND_LIBRARIES}
${catkin_LIBRARIES}
${orocos_kdl_LIBRARIES}
${kdl_parser_LIBRARIES}
addverb_cobot_hw
)
target_include_directories(ros_wrapper PUBLIC include/addverb_cobot_control)

####################### TEST Scripts

# Action Client
add_executable(test_ros_wrapper src/test_ros_wrapper.cpp)
target_link_libraries(test_ros_wrapper 
${COBOT_UI_BACKEND_LIBRARIES}
${catkin_LIBRARIES}
${orocos_kdl_LIBRARIES}
${kdl_parser_LIBRARIES}
)
target_include_directories(test_ros_wrapper PUBLIC include/addverb_cobot_control ${COBOT_UI_BACKEND_INCLUDE_DIRS})
