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

project(CuOmmBaking LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(BuildConfig)
include(CTest)

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 11.1 REQUIRED )
endif()

# Set OptiX_INSTALL_DIR to the root of the OptiX SDK when configuring CMake.
set(OptiX_INSTALL_DIR "OptiX_INSTALL_DIR-NOTFOUND" CACHE PATH "Path to OptiX installed location.")
find_package(OptiX 7.6)
if(NOT OptiX_FOUND OR OPTIX_VERSION VERSION_LESS 7.6)
  message(WARNING "Cannot build OmmBaking: it requires OptiX 7.6 or later")
  return()
endif()

option( BUILD_SHARED_LIBS "Build using shared libraries" ON )

otk_add_library( CuOmmBaking
  src/CuOmmBakingImpl.cpp
  src/CuOmmBakingImpl.cu
  src/CuOmmBakingImpl.h
  src/Evaluate.h
  src/Texture.cu
  src/Texture.h
  src/Triangle.h
  src/SummedAreaTable.h
  src/Util/BufferLayout.h
  src/Util/Exception.h
  src/Util/Rasterize.h
  src/Util/VecMath.h
  src/Util/XXH.h
)

source_group( "Header Files\\Implementation" FILES
  src/CuOmmBakingImpl.h
  src/Evaluate.h
  src/Texture.h
  src/Triangle.h
  src/SummedAreaTable.h
  src/Util/BufferLayout.h
  src/Util/Exception.h
  src/Util/Rasterize.h
  src/Util/VecMath.h
  src/Util/XXH.h
)

add_library(OptiXToolkit::CuOmmBaking ALIAS CuOmmBaking)

target_sources(CuOmmBaking
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/CuOmmBaking/CuOmmBaking.h
  include/OptiXToolkit/CuOmmBaking/CuBuffer.h
)

target_include_directories( CuOmmBaking
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${OptiX_INCLUDE_DIR}
  PRIVATE
  src
  )

target_link_libraries( CuOmmBaking
  PUBLIC
  CUDA::cudart_static
  ${CMAKE_DL_LIBS}  
  )

set_target_properties(CuOmmBaking PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON FOLDER OmmBaking)

# NVTX Profiling
option( OTK_OMM_BAKING_USE_NVTX "Enable NVTX profiling" OFF )
if( OTK_OMM_BAKING_USE_NVTX )
  target_compile_definitions( CuOmmBaking PUBLIC ENABLE_NVTX_PROFILING )
endif()

install(TARGETS CuOmmBaking
  EXPORT CuOmmBakingTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(EXPORT CuOmmBakingTargets
  FILE CuOmmBakingTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::
  )

if( BUILD_TESTING )
  add_subdirectory( tests )
  add_subdirectory( support )
endif()

if( PROJECT_IS_TOP_LEVEL )
  set( OTK_BUILD_DOCS ON CACHE BOOL "Enable build of OptiXToolkit documentation" )
  if( OTK_BUILD_DOCS )
    add_subdirectory( docs/API )
  endif()
endif()
