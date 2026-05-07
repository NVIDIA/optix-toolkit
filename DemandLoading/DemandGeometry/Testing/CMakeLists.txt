# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include( FetchGtest )

add_library( DemandGeometryTesting INTERFACE
    include/OptiXToolkit/DemandGeometry/Testing/MockGeometryLoader.h
)

target_include_directories( DemandGeometryTesting INTERFACE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries( DemandGeometryTesting INTERFACE DemandGeometry GTest::gmock )
set_target_properties( DemandGeometryTesting PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER DemandLoading
)

add_library( OptiXToolkit::DemandGeometry::Testing ALIAS DemandGeometryTesting )

target_sources(DemandGeometryTesting
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/DemandGeometry/Testing/MockGeometryLoader.h
)

install(TARGETS DemandGeometryTesting
  EXPORT DemandGeometryTestingTargets
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT DemandGeometryTestingTargets
  FILE DemandGeometryTestingTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::DemandGeometry::
)
