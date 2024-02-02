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

# We rely on CMake 3.24 for its integration of FetchContent and find_package.
# We rely on CMake 3.27 for proper CUDA compilation to optixir.
# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
include(Policies)

#########################################################
# vcpkg configuration (must precede project())

# The use of vcpkg is disabled by default if OTK is a submodule of another project,
# since the vcpkg.json manifest is written to the project root directory.
if( PROJECT_IS_TOP_LEVEL )
  option( OTK_USE_VCPKG "Use vcpkg for third party libraries" ON )
else()
  option( OTK_USE_VCPKG "Use vcpkg for third party libraries" OFF )
endif()

option(OTK_USE_VCPKG_OPENEXR "Use vcpkg to obtain OpenEXR" ${OTK_USE_VCPKG} )

 # Configure vcpkg manifest file, possibly adding OpenEXR and OpenImageIO.
if( OTK_USE_VCPKG )
  if( OTK_USE_VCPKG_OPENEXR )
    set( OTK_DEPEND_IMATH "\"imath\"," )
    set( OTK_DEPEND_OPENEXR "\"openexr\"," )
  endif()

  # OpenImageIO is too costly to include by default (it depends on Boost).
  option( OTK_FETCH_OIIO "Fetch OpenImageIO to allow DemandLoading to read PNGs and JPGs" OFF )
  if( OTK_FETCH_OIIO )
    set( OTK_DEPEND_OIIO "\"openimageio\"," )
    set( OTK_USE_OIIO ON )
  endif()

  configure_file( vcpkg.json.in ${PROJECT_SOURCE_DIR}/vcpkg.json @ONLY )

  # Load vcpkg toolchain file, which triggers downloading and building dependencies in the manifest.
  if(NOT CMAKE_TOOLCHAIN_FILE)
    include( vcpkg/scripts/buildsystems/vcpkg.cmake )
  endif()
endif()

#########################################################
# Set the project name (i.e. the VS solution file).
project(OptiXToolkit LANGUAGES C CXX CUDA VERSION 1.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

include(BuildConfig)
include(CTest)

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
# Third-party libraries

find_package(CUDAToolkit)

# Set OptiX_INSTALL_DIR to the root of the OptiX SDK when configuring CMake.
set(OptiX_INSTALL_DIR "OptiX_INSTALL_DIR-NOTFOUND" CACHE PATH "Path to OptiX installed location.")
find_package(OptiX)


#########################################################
# Subdirectories

if(PROJECT_IS_TOP_LEVEL)
  option( OTK_BUILD_EXAMPLES "Enable build of OptiXToolkit examples" ON)
  option( OTK_BUILD_DOCS "Enable build of OptiXToolkit documentation" ON)
  option( OTK_BUILD_PYOPTIX "Enable build of PyOptiX libraries" ON)
else()
  option( OTK_BUILD_EXAMPLES "Enable build of OptiXToolkit examples" OFF)
  option( OTK_BUILD_DOCS "Enable build of OptiXToolkit documentation" OFF)
  option( OTK_BUILD_PYOPTIX "Enable build of PyOptiX libraries" OFF)
endif()

function(addOptionalSubdirectory dir)
  if(EXISTS ${PROJECT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endfunction()

addOptionalSubdirectory(OmmBaking)
addOptionalSubdirectory(Memory)
addOptionalSubdirectory(DemandLoading)
addOptionalSubdirectory(ShaderUtil)

if( UNIX AND OTK_BUILD_PYOPTIX )
  addOptionalSubdirectory(PyOptiX)
endif()

if( OTK_BUILD_EXAMPLES )
   addOptionalSubdirectory(examples)
endif()

if( OTK_BUILD_DOCS )
  addOptionalSubdirectory( docs/API )
endif()
