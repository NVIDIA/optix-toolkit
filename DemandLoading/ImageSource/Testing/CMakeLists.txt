# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include( FetchGtest )

add_library( ImageSourceTesting INTERFACE
    include/OptiXToolkit/ImageSource/Testing/MockImageSource.h
)

target_include_directories( ImageSourceTesting INTERFACE include )
target_link_libraries( ImageSourceTesting INTERFACE ImageSource GTest::gmock )
set_target_properties( ImageSourceTesting PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER DemandLoading
)

add_library( OptiXToolkit::ImageSource::Testing ALIAS ImageSourceTesting )
