# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(FetchGtest)
include(GoogleTest)

add_executable(TestPbrtSceneLoader
    TestPbrtApi.cpp
)
target_link_libraries(TestPbrtSceneLoader PUBLIC PbrtSceneLoader GTest::gmock_main)
set_target_properties(TestPbrtSceneLoader PROPERTIES FOLDER Examples/Tests)
gtest_discover_tests(TestPbrtSceneLoader PROPERTIES LABELS DemandPbrtScene)
