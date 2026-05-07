# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(BuildConfig)
include(FetchGtest)

otk_add_library( Testing STATIC
  include/OptiXToolkit/Testing/Matchers.h
  include/OptiXToolkit/Testing/MockOptix.h
  include/OptiXToolkit/Testing/OptixCompare.h
  src/MockOptix.cpp
)
set_property(TARGET Testing PROPERTY FOLDER Testing)

add_library(OptiXToolkit::Testing ALIAS Testing)

target_sources(Testing
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/Testing/Matchers.h
  include/OptiXToolkit/Testing/MockOptix.h
  include/OptiXToolkit/Testing/OptixCompare.h
)

target_include_directories( Testing
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

# error C1128: number of sections exceeded object file format limit: compile with /bigobj
target_compile_options(Testing PUBLIC $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/bigobj>)

target_link_libraries( Testing
  PUBLIC
  OptiXToolkit::Error
  OptiXToolkit::Memory
  OptiX::OptiX
  GTest::gmock
  )

if( BUILD_TESTING )
  add_subdirectory( tests )
endif()
