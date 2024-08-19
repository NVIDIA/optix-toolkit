# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(FetchGtest)
include(GoogleTest)
include(ImageTest)

add_executable(TestDemandPbrtSceneImpl
    GeometryInstancePrinter.h
    Matchers.h
    MockDemandTextureCache.h
    MockGeometryLoader.h
    MockImageSource.h
    MockMaterialLoader.h
    MockMaterialResolver.h
    MockMeshLoader.h
    MockProgramGroups.h
    MockRenderer.h
    NullCast.h
    ParamsPrinters.h
    TestDemandTextureCache.cpp
    TestGeometryCache.cpp
    TestGeometryResolver.cpp
    TestMatchers.cpp
    TestMaterialResolver.cpp
    TestOptions.cpp
    TestOptixRenderer.cpp
    TestParams.cpp
    TestPbrtAlphaMapImageSource.cpp
    TestPbrtScene.cpp
    TestProgramGroups.cpp
    TestIdRangePrinter.cpp
    TestSceneAdapters.cpp
    TestSceneProxy.cpp
    TestTimer.cpp
)
if(MSVC)
    # error C1128: number of sections exceeded object file format limit: compile with /bigobj
    target_compile_options(TestDemandPbrtSceneImpl PUBLIC /bigobj)
endif()
target_link_libraries(TestDemandPbrtSceneImpl PUBLIC
    DemandPbrtSceneImpl
    OptiXToolkit::DemandGeometry::Mocks
    OptiXToolkit::ImageSource
    OptiXToolkit::Memory
    OptiXToolkit::ShaderUtil
    GTest::gmock_main)
set_target_properties(TestDemandPbrtSceneImpl PROPERTIES FOLDER Examples/DemandLoading/Tests)
gtest_discover_tests(TestDemandPbrtSceneImpl PROPERTIES LABELS DemandPbrtScene)

add_subdirectory(images)
