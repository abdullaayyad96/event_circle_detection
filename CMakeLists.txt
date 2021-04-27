cmake_minimum_required(VERSION 2.8.3)
project(event_circle_tracker)

find_package(catkin_simple REQUIRED)

catkin_simple()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(Boost REQUIRED COMPONENTS system thread)

# make the executable


# link the executable to the necesarry libs

# Install the nodelet library
cs_install()
