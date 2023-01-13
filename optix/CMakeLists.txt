# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 3.17)
project(optix)

#------------------------------------------------------------------------------
# Fetch external dependencies
#------------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.9.2
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

#------------------------------------------------------------------------------
# set environment
#------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

find_package( OptiX REQUIRED     )
find_package( CUDAToolkit 10.0 REQUIRED )

configure_file("${CMAKE_SOURCE_DIR}/path_util.py.in" "${CMAKE_SOURCE_DIR}/../examples/path_util.py")


#------------------------------------------------------------------------------
# main build
#------------------------------------------------------------------------------

pybind11_add_module(optix main.cpp)

target_link_libraries( optix PRIVATE
  OptiX::OptiX
  CUDA::cuda_driver
	)
target_compile_features( optix PRIVATE
    cxx_std_17
    )
add_custom_command(
    TARGET optix POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${OptiX_INCLUDE_DIR}
    $<TARGET_FILE_DIR:optix>/include
    )

