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
include(../../../CMake/Policies.cmake)

project(DemandLoadTexture LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(../../../CMake/BuildConfig.cmake)
include(../../../CMake/FetchGlad.cmake)
include(../../../CMake/embed_ptx.cmake)

if(PROJECT_IS_TOP_LEVEL)
  find_package(OptiXToolkit REQUIRED)
endif()

embed_ptx(
  OUTPUT_TARGET
    TextureKernel
  PTX_LINK_LIBRARIES
    OptiXToolkit::DemandLoading OptiXToolkit::Cuda
  SOURCES
    textureKernel.cu
)

add_executable( demandLoadTexture
  texture.cpp
  textureKernel.h
  )

target_link_libraries( demandLoadTexture
  TextureKernel
  OptiXToolkit::DemandLoading
  OptiXToolkit::Gui
  OptiXToolkit::Util
  glad
  )

set_target_properties( demandLoadTexture PROPERTIES INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

get_target_property( _bin demandLoadTexture RUNTIME_OUTPUT_DIRECTORY )
file( INSTALL Textures DESTINATION ${_bin} )

install( TARGETS demandLoadTexture RUNTIME )
install( DIRECTORY Textures TYPE BIN )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:demandLoadTexture> )
  add_custom_command( TARGET demandLoadTexture POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:demandLoadTexture> $<TARGET_FILE_DIR:demandLoadTexture>
    COMMAND_EXPAND_LISTS )
endif()

# Install shared libraries that the installed executable depends on.
install( PROGRAMS $<TARGET_RUNTIME_DLLS:demandLoadTexture>
         DESTINATION ${CMAKE_INSTALL_BINDIR} )
