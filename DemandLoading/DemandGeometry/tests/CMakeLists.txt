# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(CTest)
include(FetchGtest)
include(GoogleTest)

source_group( "CUDA Files" REGULAR_EXPRESSION ".+\\.cu$" )

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
target_link_libraries( TestDemandGeometry PRIVATE
    OptiXToolkit::DemandGeometry
    OptiXToolkit::DemandLoading::Testing
    OptiXToolkit::Error
    OptiXToolkit::ShaderUtil
    OptiXToolkit::Testing
    GTest::gmock_main )
set_target_properties(TestDemandGeometry PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER DemandLoading/Tests
  )
# Single CTest entry per binary: this binary creates a CUDA/OptiX context that
# is reused across cases, so one process pays context init once, not per case.
add_test( NAME TestDemandGeometry COMMAND TestDemandGeometry )
set_tests_properties( TestDemandGeometry PROPERTIES LABELS DemandLoading )
