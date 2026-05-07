# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(CTest)
include(FetchGtest)
include(GoogleTest)

source_group( "CUDA Files" REGULAR_EXPRESSION ".+\\.cu$" )

otk_add_library(DemandGeometryMocks INTERFACE
  include/OptiXToolkit/DemandGeometry/Mocks/MockDemandLoader.h
)
target_include_directories(DemandGeometryMocks INTERFACE include)
target_link_libraries(DemandGeometryMocks INTERFACE OptiXToolkit::Testing OptiX::OptiX GTest::gmock CUDA::cuda_driver)
set_target_properties(DemandGeometryMocks PROPERTIES FOLDER DemandLoading/Tests)
add_library(OptiXToolkit::DemandGeometry::Mocks ALIAS DemandGeometryMocks)

otk_add_executable( TestDemandGeometry
  LaunchIntersectAabb.h
  LaunchIntersectAabb.cu
  TestIntersectAabb.cpp
  TestProxyInstances.cpp
)
if(MSVC)
  # Disable unused formal parameter warning in gmock.
  target_compile_options(TestDemandGeometry PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/wd4100>)
endif()
target_link_libraries( TestDemandGeometry PRIVATE OptiXToolkit::DemandGeometry::Mocks OptiXToolkit::Error OptiXToolkit::ShaderUtil OptiXToolkit::DemandGeometry GTest::gmock_main )
set_target_properties(TestDemandGeometry PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER DemandLoading/Tests
  )
gtest_discover_tests(TestDemandGeometry PROPERTIES LABELS DemandLoading)
