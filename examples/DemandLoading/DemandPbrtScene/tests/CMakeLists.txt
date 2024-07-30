#
#  Copyright (c) 2023 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

include(FetchGtest)
include(GoogleTest)
include(ImageTest)

add_executable(TestDemandPbrtSceneImpl
    Matchers.h
    MockGeometryLoader.h
    MockImageSource.h
    NullCast.h
    ParamsPrinters.h
    TestDemandTextureCache.cpp
    TestGeometryCache.cpp
    TestMatchers.cpp
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
