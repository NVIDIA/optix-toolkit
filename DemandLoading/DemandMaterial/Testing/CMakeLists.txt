# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include( FetchGtest )

add_library( DemandMaterialTesting INTERFACE
    include/OptiXToolkit/DemandMaterial/Testing/MockMaterialLoader.h
)

target_include_directories( DemandMaterialTesting INTERFACE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries( DemandMaterialTesting INTERFACE DemandMaterial GTest::gmock )
set_target_properties( DemandMaterialTesting PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER DemandLoading
)

add_library( OptiXToolkit::DemandMaterial::Testing ALIAS DemandMaterialTesting )

target_sources(DemandMaterialTesting
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/DemandMaterial/Testing/MockMaterialLoader.h
)

install(TARGETS DemandMaterialTesting
  EXPORT DemandMaterialTestingTargets
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT DemandMaterialTestingTargets
  FILE DemandMaterialTestingTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::DemandMaterial::
)
