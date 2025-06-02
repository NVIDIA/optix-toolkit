# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
include(Policies)
include(SetCxxStandard)

project(ShaderUtil LANGUAGES C CXX CUDA)
set_cxx_standard(11)
set(CMAKE_CXX_EXTENSIONS OFF)

include(BuildConfig)

if(MSVC)
  add_definitions(-DNOMINMAX)
endif()

# Use fast math for CUDA code.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 10.0 REQUIRED )
endif()

add_library(ShaderUtil INTERFACE)
add_library(OptiXToolkit::ShaderUtil ALIAS ShaderUtil)

target_sources(ShaderUtil
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/ShaderUtil/AliasTable.h
  include/OptiXToolkit/ShaderUtil/CdfInversionTable.h
  include/OptiXToolkit/ShaderUtil/color.h
  include/OptiXToolkit/ShaderUtil/CudaSelfIntersectionAvoidance.h
  include/OptiXToolkit/ShaderUtil/DebugLocation.h
  include/OptiXToolkit/ShaderUtil/OptixSelfIntersectionAvoidance.h
  include/OptiXToolkit/ShaderUtil/PdfTable.h
  include/OptiXToolkit/ShaderUtil/Preprocessor.h
  include/OptiXToolkit/ShaderUtil/ray_cone.h
  include/OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance.h
  include/OptiXToolkit/ShaderUtil/SelfIntersectionAvoidanceTypes.h
  include/OptiXToolkit/ShaderUtil/ISummedAreaTable.h
  include/OptiXToolkit/ShaderUtil/Transform4.h
  include/OptiXToolkit/ShaderUtil/Transform4Printer.h
  include/OptiXToolkit/ShaderUtil/vec_math.h
  include/OptiXToolkit/ShaderUtil/vec_printers.h
  )

target_include_directories(ShaderUtil INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

set_target_properties(ShaderUtil PROPERTIES FOLDER "ShaderUtil")

install(TARGETS ShaderUtil
  EXPORT ShaderUtilTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
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
