# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
include(Policies)
include(SetCxxStandard)

project(CuOmmBaking LANGUAGES C CXX CUDA)
set_cxx_standard(17) # CUB requires C++17
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(BuildConfig)
include(CTest)
include(FetchOptiX)

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 11.1 REQUIRED )
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
