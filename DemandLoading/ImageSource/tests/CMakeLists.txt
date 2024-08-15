# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include( FetchGtest )
include( GoogleTest )

# Embed CMAKE_CURRENT_SOURCE_DIR in getSourceDir()
configure_file( ImageSourceTestConfig.h.in include/ImageSourceTestConfig.h @ONLY )

otk_add_executable( testImageSource
  MockImageSource.h
  TestCheckerBoardImage.cpp
  TestImageSourceCache.cpp
  TestMipMapImageSource.cpp
  TestTiledImageSource.cpp
  ImageSourceTestConfig.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/ImageSourceTestConfig.h
)
if(OTK_USE_OIIO OR OTK_USE_OPENEXR)
    target_sources(testImageSource PUBLIC TestImageSource.cpp)
endif()
if(OTK_USE_OPENEXR)
    target_sources(testImageSource PUBLIC TestRateLimitedImageSource.cpp)
endif()
source_group("CMake Templates" REGULAR_EXPRESSION ".*\.in$")

target_include_directories( testImageSource PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include )
target_link_libraries( testImageSource PUBLIC
    ImageSource
    DemandLoading
    OptiXToolkit::ShaderUtil
    OpenEXR::OpenEXR # for half
    GTest::gmock_main )
set_target_properties( testImageSource PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER DemandLoading/Tests
)

# Register test cases with CTest.
gtest_discover_tests(testImageSource PROPERTIES LABELS ImageSource)
