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

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
include(Policies)

project(ShaderUtil LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(BuildConfig)

find_package( CUDAToolkit 10.0 REQUIRED )

add_library(ShaderUtil INTERFACE)
add_library(OptiXToolkit::ShaderUtil ALIAS ShaderUtil)

target_sources(ShaderUtil
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/ShaderUtil/color.h
  include/OptiXToolkit/ShaderUtil/Preprocessor.h
  include/OptiXToolkit/ShaderUtil/vec_math.h
  )

target_include_directories(ShaderUtil INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

install(TARGETS ShaderUtil
  EXPORT ShaderUtilTargets
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(EXPORT ShaderUtilTargets
  FILE ShaderUtilTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::
  )

if( BUILD_TESTING )
  add_subdirectory( tests )
endif()
