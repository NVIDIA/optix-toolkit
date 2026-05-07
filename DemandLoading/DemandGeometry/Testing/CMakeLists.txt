# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include( FetchGtest )

add_library( DemandGeometryTesting INTERFACE
    include/OptiXToolkit/DemandGeometry/Testing/MockGeometryLoader.h
)

target_include_directories( DemandGeometryTesting INTERFACE include )
target_link_libraries( DemandGeometryTesting INTERFACE DemandGeometry GTest::gmock )
set_target_properties( DemandGeometryTesting PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER DemandLoading
)

add_library( OptiXToolkit::DemandGeometry::Testing ALIAS DemandGeometryTesting )
