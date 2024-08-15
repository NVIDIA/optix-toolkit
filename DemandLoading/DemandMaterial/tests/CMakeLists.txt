# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(CTest)
include(FetchGtest)
include(GoogleTest)

otk_add_executable( TestDemandMaterial
    TestMaterialLoader.cpp
)
target_link_libraries( TestDemandMaterial
    PRIVATE
        OptiXToolkit::Error
        OptiXToolkit::DemandGeometry::Mocks
        OptiXToolkit::DemandMaterial
        GTest::gmock_main )
set_target_properties( TestDemandMaterial PROPERTIES 
    CXX_STANDARD 14  # Required by latest gtest
    FOLDER DemandLoading/Tests
)
gtest_discover_tests( TestDemandMaterial PROPERTIES LABELS DemandLoading )
