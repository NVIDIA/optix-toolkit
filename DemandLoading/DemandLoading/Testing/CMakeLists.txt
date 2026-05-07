# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include( FetchGtest )

add_library( DemandLoadingTesting INTERFACE
    include/OptiXToolkit/DemandLoading/Testing/MockDemandLoader.h
)

target_include_directories( DemandLoadingTesting INTERFACE include )
target_link_libraries( DemandLoadingTesting INTERFACE DemandLoading GTest::gmock )
set_target_properties( DemandLoadingTesting PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER DemandLoading
)

add_library( OptiXToolkit::DemandLoading::Testing ALIAS DemandLoadingTesting )
