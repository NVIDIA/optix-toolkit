# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include( FetchGtest )

add_library( DemandMaterialTesting INTERFACE
    include/OptiXToolkit/DemandMaterial/Testing/MockMaterialLoader.h
)

target_include_directories( DemandMaterialTesting INTERFACE include )
target_link_libraries( DemandMaterialTesting INTERFACE DemandMaterial GTest::gmock )
set_target_properties( DemandMaterialTesting PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER DemandLoading
)

add_library( OptiXToolkit::DemandMaterial::Testing ALIAS DemandMaterialTesting )
