# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(CTest)
include(FetchGtest)
include(GoogleTest)

source_group( "CUDA Files" REGULAR_EXPRESSION ".+\\.cu$" )

otk_add_library(DemandGeometryMocks STATIC
  include/OptiXToolkit/DemandGeometry/Mocks/Matchers.h
  include/OptiXToolkit/DemandGeometry/Mocks/MockDemandLoader.h
  include/OptiXToolkit/DemandGeometry/Mocks/MockOptix.h
  include/OptiXToolkit/DemandGeometry/Mocks/OptixCompare.h
  MockOptix.cpp
)
if(MSVC)
  # error C1128: number of sections exceeded object file format limit: compile with /bigobj
  target_compile_options(DemandGeometryMocks PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()
target_include_directories(DemandGeometryMocks PUBLIC include)
target_link_libraries(DemandGeometryMocks PUBLIC OptiXToolkit::Error OptiXToolkit::Memory OptiX::OptiX GTest::gmock CUDA::cuda_driver)
set_target_properties(DemandGeometryMocks PROPERTIES FOLDER DemandLoading/Tests)
add_library(OptiXToolkit::DemandGeometry::Mocks ALIAS DemandGeometryMocks)

otk_add_executable(TestDemandGeometryMocks
  TestMatchers.cpp
  TestOptixCompare.cpp
)
if(MSVC)
  # Disable unused formal parameter warning in gmock.
  target_compile_options(TestDemandGeometryMocks PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/wd4100>)
endif()
target_link_libraries( TestDemandGeometryMocks PRIVATE OptiXToolkit::DemandGeometry::Mocks OptiXToolkit::Error OptiXToolkit::ShaderUtil OptiXToolkit::DemandGeometry GTest::gmock_main )
set_target_properties(TestDemandGeometryMocks PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER DemandLoading/Tests
  )
gtest_discover_tests(TestDemandGeometryMocks PROPERTIES LABELS DemandLoading)

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
