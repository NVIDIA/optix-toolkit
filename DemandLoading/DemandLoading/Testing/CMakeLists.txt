# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include( FetchGtest )

add_library( DemandLoadingTesting INTERFACE
    include/OptiXToolkit/DemandLoading/Testing/MockDemandLoader.h
)

target_include_directories( DemandLoadingTesting INTERFACE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries( DemandLoadingTesting INTERFACE DemandLoading GTest::gmock )
set_target_properties( DemandLoadingTesting PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER DemandLoading
)

add_library( OptiXToolkit::DemandLoading::Testing ALIAS DemandLoadingTesting )

target_sources(DemandLoadingTesting
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/DemandLoading/Testing/MockDemandLoader.h
)

install(TARGETS DemandLoadingTesting
  EXPORT DemandLoadingTestingTargets
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT DemandLoadingTestingTargets
  FILE DemandLoadingTestingTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::DemandLoading::
)
