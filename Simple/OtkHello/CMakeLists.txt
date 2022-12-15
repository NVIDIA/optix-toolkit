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

project(OtkHello LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(BuildConfig)
include(FetchGlad)
include(embed_ptx)

find_package( CUDAToolkit 11.0 REQUIRED )

# Set OptiX_INSTALL_DIR to the root of the OptiX SDK when configuring CMake.
set(OptiX_INSTALL_DIR "OptiX_INSTALL_DIR-NOTFOUND" CACHE PATH "Path to OptiX installed location.")
list(APPEND CMAKE_MODULE_PATH ${OTK_SOURCE_DIR}/CMake)
find_package(OptiX REQUIRED)

if(PROJECT_IS_TOP_LEVEL)
  find_package(OptiXToolkit REQUIRED)
endif()

embed_ptx(
  OUTPUT_TARGET
    DrawSolidColor
  PTX_LINK_LIBRARIES
    OptiXToolkit::ShaderUtil
    OptiX::OptiX
  SOURCES
    draw_solid_color.cu
)

add_executable( otkHello
  hello.cpp
  hello.h
  )

target_link_libraries( otkHello
  DrawSolidColor
  OptiXToolkit::Gui
  OptiXToolkit::Util
  OptiX::OptiX
  glfw
  glad::glad
  CUDA::cuda_driver
  CUDA::cudart
  ${CMAKE_DL_LIBS}  
  )

set_target_properties( otkHello PROPERTIES INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:otkHello> )
  add_custom_command( TARGET otkHello POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:otkHello> $<TARGET_FILE_DIR:otkHello>
    COMMAND_EXPAND_LISTS )
endif()
