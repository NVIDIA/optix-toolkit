# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 2.8.12)
project(optix)

set(CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/CMake"
    ${CMAKE_MODULE_PATH}
    )

find_package( OptiX REQUIRED     )
find_package( CUDA 10.0 REQUIRED )

include_directories(
    "${OptiX_INCLUDE}"
    "${CUDA_INCLUDE_DIRS}"
    )

configure_file("${CMAKE_SOURCE_DIR}/path_util.py.in" "${CMAKE_SOURCE_DIR}/../examples/path_util.py")

add_subdirectory(pybind11)

pybind11_add_module(optix main.cpp)
target_link_libraries( optix PRIVATE
	"${CUDA_LIBRARIES}"
	)
target_compile_features( optix PRIVATE 
    cxx_std_17
    )

add_custom_command(
    TARGET optix POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${OptiX_INCLUDE}
    $<TARGET_FILE_DIR:optix>/include
)

