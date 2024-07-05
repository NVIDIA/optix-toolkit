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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(BuildConfig)
include(embed_cuda)

find_package(OptiX 7.5)
if(NOT OptiX_FOUND OR OPTIX_VERSION VERSION_LESS 7.5)
    message(WARNING "Cannot build DemandPbrtScene: it requires OptiX 7.5 or greater.")
    return()
endif()

find_package(CUDAToolkit 11.1 REQUIRED)

embed_cuda(CONST HEADER DemandPbrtSceneKernelCuda.h
    OUTPUT_TARGET DemandPbrtSceneKernel
    LIBRARIES
        OptiXToolkit::DemandGeometry
        OptiXToolkit::DemandLoading
        OptiXToolkit::DemandMaterial
        OptiXToolkit::ImageSource
        OptiXToolkit::ShaderUtil
    SOURCES
        DemandPbrtScene.cu
        PhongMaterial.cu
    FOLDER Examples/DemandLoading
)
target_sources( DemandPbrtSceneKernelCuda
    PRIVATE
        DeviceTriangles.h
        PhongShade.h
)

add_library(DemandPbrtSceneImpl STATIC
    Accumulator.h
    Application.h
    Application.cpp
    CudaContext.h
    CudaContext.cpp
    DemandTextureCache.h
    DemandTextureCache.cpp
    Dependencies.h
    FrameRate.h
    GeometryCache.h
    GeometryCache.cpp
    GeometryCacheStatistics.h
    IdRangePrinter.h
    ImageSourceFactory.h
    ImageSourceFactory.cpp
    ImageSourceFactoryStatistics.h
    ImGuiUserInterface.cpp
    ImGuiUserInterface.h
    Options.h
    Options.cpp
    OptixRenderer.h
    OptixRenderer.cpp
    Params.h
    PbrtAlphaMapImageSource.h
    PbrtAlphaMapImageSource.cpp
    PbrtScene.h
    PbrtScene.cpp
    Primitive.h
    ProxyFactoryStatistics.h
    Renderer.h
    Sample.h
    Scene.h
    SceneAdapters.h
    SceneProxy.h
    SceneProxy.cpp
    SceneStatistics.h
    Statistics.cpp
    Statistics.h
    Timer.h
    Traversable.h
    UserInterface.h
    UserInterfaceStatistics.h
)
set_target_properties(DemandPbrtSceneImpl PROPERTIES FOLDER "Examples/DemandLoading")
target_include_directories(DemandPbrtSceneImpl PUBLIC .)
target_link_libraries(DemandPbrtSceneImpl PUBLIC 
    PbrtSceneLoader
    DemandPbrtSceneKernel
    OptiXToolkit::DemandMaterial
    OptiXToolkit::DemandGeometry
    OptiXToolkit::Error
    OptiXToolkit::Gui
    OptiXToolkit::Memory
    OptiXToolkit::OptiXMemory
    OptiXToolkit::Util
    OptiXToolkit::ShaderUtil
    OptiX::OptiX
)

add_executable(DemandPbrtScene main.cpp ReadMe.md)
target_link_libraries(DemandPbrtScene PUBLIC DemandPbrtSceneImpl)
set_target_properties(DemandPbrtScene PROPERTIES FOLDER "Examples/DemandLoading")

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
