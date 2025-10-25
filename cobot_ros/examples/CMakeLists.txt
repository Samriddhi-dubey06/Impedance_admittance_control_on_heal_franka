cmake_minimum_required(VERSION 3.5)
project(examples)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  std_msgs
  trajectory_msgs
)


catkin_package(
#  INCLUDE_DIRS include
 CATKIN_DEPENDS geometry_msgs roscpp std_msgs trajectory_msgs
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)


# DEMO TWIST
add_executable(demo_twist src/demo_twist.cpp)
target_link_libraries(demo_twist
${catkin_LIBRARIES}
)
target_include_directories(demo_twist PUBLIC include/addverb_cobot_control)

# DEMO VELOCITY
add_executable(demo_velocity src/demo_velocity.cpp)
target_link_libraries(demo_velocity
${catkin_LIBRARIES}
)
target_include_directories(demo_velocity PUBLIC include/addverb_cobot_control)

# DEMO JOINT TRAJECTORY
add_executable(demo_joint_trajectory src/demo_joint_trajectory.cpp)
target_link_libraries(demo_joint_trajectory
${catkin_LIBRARIES}
)
target_include_directories(demo_joint_trajectory PUBLIC include/addverb_cobot_control)

# DEMO MULTI TRAJECTORY
add_executable(demo_multi_trajectory src/demo_multi_trajectory.cpp)
target_link_libraries(demo_multi_trajectory
${catkin_LIBRARIES}
)
target_include_directories(demo_multi_trajectory PUBLIC include/addverb_cobot_control)