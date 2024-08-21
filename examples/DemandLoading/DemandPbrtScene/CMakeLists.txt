# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

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

option(DEMANDPBRTSCENE_PBRT_CAMERA "Enable interactive switch to pbrt camera model (WIP)" OFF)
mark_as_advanced(DEMANDPBRTSCENE_PBRT_CAMERA)
configure_file(Config.h.in Config.h)

add_library(DemandPbrtSceneImpl STATIC
    Accumulator.h
    Application.h
    Application.cpp
    Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Config.h
    CudaContext.h
    CudaContext.cpp
    DemandTextureCache.h
    DemandTextureCache.cpp
    DemandTextureCacheStatistics.h
    Dependencies.h
    FrameRate.h
    FrameStopwatch.h
    GeometryCache.h
    GeometryCache.cpp
    GeometryCacheStatistics.h
    GeometryResolver.h
    GeometryResolver.cpp
    GeometryResolverStatistics.h
    IdRangePrinter.h
    ImageSourceFactory.h
    ImageSourceFactory.cpp
    ImageSourceFactoryStatistics.h
    ImGuiUserInterface.cpp
    ImGuiUserInterface.h
    MaterialBatch.h
    MaterialBatch.cpp
    MaterialResolver.h
    MaterialResolver.cpp
    MaterialResolverStatistics.h
    Options.h
    Options.cpp
    OptixRenderer.h
    OptixRenderer.cpp
    Params.h
    PbrtAlphaMapImageSource.h
    PbrtAlphaMapImageSource.cpp
    PbrtScene.cpp
    Primitive.h
    ProgramGroups.h
    ProgramGroups.cpp
    ProxyFactoryStatistics.h
    ProxyGranularity.h
    RenderMode.h
    Renderer.h
    Sample.h
    Scene.h
    SceneAdapters.h
    SceneProxy.h
    SceneProxy.cpp
    SceneStatistics.h
    SceneSyncState.h
    Stopwatch.h
    Statistics.cpp
    Statistics.h
    Timer.h
    UserInterface.h
    UserInterfaceStatistics.h
)
source_group("CMake Template Files" REGULAR_EXPRESSION "^.*\.in$")
set_target_properties(DemandPbrtSceneImpl PROPERTIES FOLDER "Examples/DemandLoading")
target_include_directories(DemandPbrtSceneImpl PUBLIC .)
target_include_directories(DemandPbrtSceneImpl PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
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
