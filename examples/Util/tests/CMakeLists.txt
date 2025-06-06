# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include( FetchGtest )
include( FetchOptiX )
include( GoogleTest )

otk_add_executable( testUtil
    TestEXRInputFile.cpp
    FunctionTable.cpp
    SourceDir.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/SourceDir.h
  )
set_target_properties(testUtil PROPERTIES FOLDER Examples/Tests)


target_include_directories( testUtil PUBLIC
  ../src
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>  # for SourceDir.h
  )

# Embed CMAKE_CURRENT_SOURCE_DIR in getSourceDir()
configure_file( SourceDir.h.in include/SourceDir.h @ONLY )

set_target_properties(testUtil PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER Examples/Tests 
)

target_link_libraries( testUtil
  Util
  GTest::gtest_main
  Imath::Imath
  OptiX::OptiX
  Stb::Image
  )

# Register test cases with CTest.
gtest_discover_tests(testUtil PROPERTIES LABELS examples)
