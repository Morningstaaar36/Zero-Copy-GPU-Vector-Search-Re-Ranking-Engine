cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(zcgvs LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(ZCGVS_BUILD_PYTHON "Build the pybind11 module" ON)
option(ZCGVS_BUILD_APPS "Build the C++ CLI and kernel benchmark" ON)
option(ZCGVS_BUILD_TESTS "Build the CUDA correctness harnesses" ON)
option(ZCGVS_ENABLE_NVTX "Emit NVTX ranges for Nsight Systems" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(ONNXRuntime REQUIRED)

add_subdirectory(src)

if(ZCGVS_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

if(ZCGVS_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(ZCGVS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "zcgvs: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "zcgvs: ONNX Runtime       = ${ONNXRuntime_LIBRARY}")
message(STATUS "zcgvs: NVTX ranges        = ${ZCGVS_ENABLE_NVTX}")
