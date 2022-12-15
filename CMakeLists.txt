#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#########################################################
# Welcome to the OptiX Toolkit (OTK)

# If you have any questions, we encourage you to post on the OptiX forums:
# https://devtalk.nvidia.com/default/board/90/

# CMake helper files are located in the CMake subdirectory.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
include(Policies)

#########################################################
# Set the project name (i.e. the VS solution file).
project(OptiXToolkit LANGUAGES C CXX CUDA VERSION 1.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

include(BuildConfig)

#########################################################
# Create and install package configuration and version files.

include(GNUInstallDirs)

configure_file(
  ${OptiXToolkit_SOURCE_DIR}/CMake/OptiXToolkitConfig.cmake.in
  ${OptiXToolkit_BINARY_DIR}/CMake/OptiXToolkitConfig.cmake @ONLY)

configure_file(
  ${OptiXToolkit_SOURCE_DIR}/CMake/OptiXToolkitConfigVersion.cmake.in
  ${OptiXToolkit_BINARY_DIR}/CMake/OptiXToolkitConfigVersion.cmake @ONLY)

install(FILES ${OptiXToolkit_BINARY_DIR}/CMake/OptiXToolkitConfig.cmake
  ${OptiXToolkit_BINARY_DIR}/CMake/OptiXToolkitConfigVersion.cmake
  CMake/FindOptiX.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit)

#########################################################
# Subdirectories

set(OTK_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "OptiX Toolkit source directory")

if(PROJECT_IS_TOP_LEVEL)
  option( OTK_BUILD_TESTS "Enable build of OptiXToolkit tests" ON)
  option( OTK_BUILD_DOCS "Enable build of OptiXToolkit documentation" ON)
else()
  option( OTK_BUILD_TESTS "Enable build of OptiXToolkit tests" OFF)
  option( OTK_BUILD_DOCS "Enable build of OptiXToolkit documentation" OFF)
endif()

add_subdirectory(CuOmmBaking)
add_subdirectory(DemandLoading)
add_subdirectory(ShaderUtil)
add_subdirectory(Util)

if( OTK_BUILD_DOCS )
  add_subdirectory( docs/API )
endif()
