
#
cmake_minimum_required(VERSION 3.1....3.29)

# 
project(
        KoolKernels
        VERSION 1.0
        LANGUAGES CXX
)

add_library(readCSV.cpp)

# cuda specific
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
