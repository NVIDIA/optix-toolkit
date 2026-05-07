# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(CTest)
include(FetchGtest)
include(GoogleTest)

otk_add_executable(TestOptixMocks
  TestMatchers.cpp
  TestMockOptix.cpp
  TestOptixCompare.cpp
)

# Disable unused formal parameter warning in gmock.
target_compile_options(TestOptixMocks PRIVATE $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/wd4100>)

target_link_libraries( TestOptixMocks PRIVATE OptiXToolkit::Testing OptiXToolkit::Memory GTest::gmock_main )
set_target_properties(TestOptixMocks PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER Testing/Tests
  )
gtest_discover_tests(TestOptixMocks PROPERTIES LABELS Testing)
