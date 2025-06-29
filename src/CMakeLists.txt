
#
cmake_minimum_required(VERSION 3.17...3.29)

# 
project(
        KoolKernels
        VERSION 1.0
        LANGUAGES CXX CUDA
)

# cuda specific
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(readCSV readCSV.cpp)

add_executable(deviceProperties deviceProperties.cpp)
target_link_libraries(deviceProperties CUDA::cudart)

add_executable(vecAddmain vecAdd/main.cu vecAdd/vecAdd.cu)
set_property(TARGET vecAddmain PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(vecAddmain readCSV CUDA::cudart)
