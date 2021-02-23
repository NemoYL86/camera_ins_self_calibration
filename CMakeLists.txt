cmake_minimum_required(VERSION 2.8.3)
project(visual_mapping)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O3 -Wall -g -fopenmp")

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        geometry_msgs
        nav_msgs
        tf
        cv_bridge
        image_transport)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include_directories(${CMAKE_SOURCE_DIR}/src
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)


##
add_executable(visual_mapping src/visual_mapping_node.cpp
        src/features/feature_tracker.cpp
        src/features/gms_rejector.cpp
        src/sfm/sfm.cpp
        src/base/kitti_parser.cpp
        )
target_link_libraries(visual_mapping ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} glog)

