cmake_minimum_required(VERSION 3.12)
project(FasttrackerCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(fasttracker STATIC
  src/basetrack.cpp
  src/fasttracker.cpp
  src/kalman_filter.cpp
  src/matching.cpp
  src/strack.cpp)

target_include_directories(fasttracker PUBLIC include)

