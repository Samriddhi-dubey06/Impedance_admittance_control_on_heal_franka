cmake_minimum_required(VERSION 3.5)
project(addverb_heal_description)

find_package(catkin REQUIRED)

catkin_package()

install(
  DIRECTORY launch mesh rviz urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

