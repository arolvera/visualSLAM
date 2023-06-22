cmake_minimum_required(VERSION 3.15)
project(visualOdometry)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})
add_executable(VO src/VO.cpp)
target_link_libraries(VO ${OpenCV_LIBS})