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

# Common CMake files are included from a top-level directory, but otherwise
# this directory can be built as a standalone project.
set(OTK_SOURCE_DIR "../../.." CACHE PATH "OptiX Toolkit source directory")

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
include(Policies)

project(DemandTextureAppBase LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(BuildConfig)
include(FetchGlfw)
include(FetchGlad)

if(PROJECT_IS_TOP_LEVEL)
  find_package(OptiXToolkit REQUIRED)
endif()

option( BUILD_SHARED_LIBS "Build using shared libraries" ON )

add_library(DemandTextureAppBase
  src/DemandTextureApp.cpp
  )

if(NOT PROJECT_IS_TOP_LEVEL)
  add_library(OptiXToolkit::DemandTextureAppBase ALIAS DemandTextureAppBase)
endif()

target_sources(DemandTextureAppBase 
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/DemandTextureAppBase/DemandTextureApp.h
  include/OptiXToolkit/DemandTextureAppBase/DemandTextureAppDeviceUtil.h
  include/OptiXToolkit/DemandTextureAppBase/LaunchParams.h
  include/OptiXToolkit/DemandTextureAppBase/PerDeviceOptixState.h
)

target_include_directories(DemandTextureAppBase PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(DemandTextureAppBase
  PUBLIC
  OptiXToolkit::DemandLoading
  OptiXToolkit::Gui
  OptiXToolkit::ShaderUtil
  OptiXToolkit::Util
  PRIVATE
  glfw
  glad::glad
  )

set_target_properties(DemandTextureAppBase PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

