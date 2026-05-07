# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include( FetchGtest )

add_library( ImageSourceTesting INTERFACE
    include/OptiXToolkit/ImageSource/Testing/MockImageSource.h
)

target_include_directories( ImageSourceTesting INTERFACE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries( ImageSourceTesting INTERFACE ImageSource GTest::gmock )
set_target_properties( ImageSourceTesting PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER DemandLoading
)

add_library( OptiXToolkit::ImageSource::Testing ALIAS ImageSourceTesting )

target_sources(ImageSourceTesting
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/ImageSource/Testing/MockImageSource.h
)

install(TARGETS ImageSourceTesting
  EXPORT ImageSourceTestingTargets
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT ImageSourceTestingTargets
  FILE ImageSourceTestingTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::ImageSource::
)
