# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(FetchGtest)
include(GoogleTest)
include(ImageTest)

add_executable(TestDemandPbrtSceneImpl
    include/DemandPbrtScene/Testing/GeometryInstancePrinter.h
    include/DemandPbrtScene/Testing/Matchers.h
    include/DemandPbrtScene/Testing/MockDemandTextureCache.h
    include/DemandPbrtScene/Testing/MockMaterialResolver.h
    include/DemandPbrtScene/Testing/MockMeshLoader.h
    include/DemandPbrtScene/Testing/MockProgramGroups.h
    include/DemandPbrtScene/Testing/MockRenderer.h
    include/DemandPbrtScene/Testing/ParamsPrinters.h
    TestDemandTextureCache.cpp
    TestGeometryCache.cpp
    TestGeometryResolver.cpp
    TestJsonStatisticsPrinter.cpp
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
target_include_directories(TestDemandPbrtSceneImpl PUBLIC include)
target_link_libraries(TestDemandPbrtSceneImpl PUBLIC
    DemandPbrtSceneImpl
    OptiXToolkit::DemandGeometry::Testing
    OptiXToolkit::DemandLoading::Testing
    OptiXToolkit::DemandMaterial::Testing
    OptiXToolkit::ImageSource::Testing
    OptiXToolkit::Memory
    OptiXToolkit::ShaderUtil
    OptiXToolkit::Testing
    GTest::gmock_main)
set_target_properties(TestDemandPbrtSceneImpl PROPERTIES FOLDER Examples/DemandLoading/Tests)
gtest_discover_tests(TestDemandPbrtSceneImpl PROPERTIES LABELS DemandPbrtScene)

add_subdirectory(images)
