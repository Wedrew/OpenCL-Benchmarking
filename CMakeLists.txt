# Required packages: sudo apt install beignet-opencl-icd mesa-opencl-icd opencl-headers

cmake_minimum_required(VERSION 3.11)
project(benchmark C)

find_package(OpenCL REQUIRED)

add_executable(benchmark benchmark.c load_kernel.h)
target_link_libraries(benchmark OpenCL::OpenCL)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/saxpy.cl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
