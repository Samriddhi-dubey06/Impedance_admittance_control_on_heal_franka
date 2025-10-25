cmake_minimum_required(VERSION 3.5)
project(addverb_cobot_msgs)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  actionlib_msgs
  trajectory_msgs
  geometry_msgs
)

## Generate messages in the 'msg' folder
# add_message_files(FILES
#   AlliedData.msg
#   Config.msg
#   ControllerEnum.msg
#   EEInertia.msg
#   TargetPosition.msg
#   RecordedPoint.msg
# )

# add_service_files(
#   FILES
#   Payload.srv
# )

add_action_files(
  FILES
  ErrorRecovery.action
  Grasp.action
  Release.action
)

generate_messages(DEPENDENCIES
  std_msgs
  actionlib_msgs
  trajectory_msgs
  geometry_msgs
)

catkin_package(CATKIN_DEPENDS
  message_runtime
  std_msgs
  trajectory_msgs
)
