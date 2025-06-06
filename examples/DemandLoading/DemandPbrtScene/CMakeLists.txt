# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(BuildConfig)
include(embed_cuda)
include(FetchOptiX)
include(SetCxxStandard)

set_cxx_standard(17)

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
    INCLUDES
        include
    FOLDER Examples/DemandLoading
)
target_sources( DemandPbrtSceneKernelCuda
    PRIVATE
        include/DemandPbrtScene/DeviceTriangles.h
        include/DemandPbrtScene/PhongShade.h
)

option(DEMANDPBRTSCENE_PBRT_CAMERA "Enable interactive switch to pbrt camera model (WIP)" OFF)
mark_as_advanced(DEMANDPBRTSCENE_PBRT_CAMERA)

configure_file(include/DemandPbrtScene/Config.h.in include/DemandPbrtScene/Config.h)

add_library(DemandPbrtSceneImpl STATIC
    Application.cpp
    CudaContext.cpp
    DemandTextureCache.cpp
    GeometryCache.cpp
    GeometryResolver.cpp
    ImageSourceFactory.cpp
    ImGuiUserInterface.cpp
    MaterialResolver.cpp
    Options.cpp
    OptixRenderer.cpp
    PbrtAlphaMapImageSource.cpp
    PbrtScene.cpp
    ProgramGroups.cpp
    SceneProxy.cpp
    Statistics.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/include/DemandPbrtScene/Config.h
    include/DemandPbrtScene/Accumulator.h
    include/DemandPbrtScene/Application.h
    include/DemandPbrtScene/Config.h.in
    include/DemandPbrtScene/Conversions.h
    include/DemandPbrtScene/CudaContext.h
    include/DemandPbrtScene/DemandTextureCache.h
    include/DemandPbrtScene/DemandTextureCacheStatistics.h
    include/DemandPbrtScene/Dependencies.h
    include/DemandPbrtScene/FrameRate.h
    include/DemandPbrtScene/FrameStopwatch.h
    include/DemandPbrtScene/GeometryCache.h
    include/DemandPbrtScene/GeometryCacheStatistics.h
    include/DemandPbrtScene/GeometryResolver.h
    include/DemandPbrtScene/GeometryResolverStatistics.h
    include/DemandPbrtScene/IdRangePrinter.h
    include/DemandPbrtScene/ImageSourceFactory.h
    include/DemandPbrtScene/ImageSourceFactoryStatistics.h
    include/DemandPbrtScene/ImGuiUserInterface.h
    include/DemandPbrtScene/JsonStatisticsPrinter.h
    include/DemandPbrtScene/MaterialAdapters.h
    include/DemandPbrtScene/MaterialResolver.h
    include/DemandPbrtScene/MaterialResolverStatistics.h
    include/DemandPbrtScene/Options.h
    include/DemandPbrtScene/OptixRenderer.h
    include/DemandPbrtScene/Params.h
    include/DemandPbrtScene/PbrtAlphaMapImageSource.h
    include/DemandPbrtScene/Primitive.h
    include/DemandPbrtScene/ProgramGroups.h
    include/DemandPbrtScene/ProxyFactoryStatistics.h
    include/DemandPbrtScene/ProxyGranularity.h
    include/DemandPbrtScene/Renderer.h
    include/DemandPbrtScene/RenderMode.h
    include/DemandPbrtScene/Sample.h
    include/DemandPbrtScene/Scene.h
    include/DemandPbrtScene/SceneAdapters.h
    include/DemandPbrtScene/SceneGeometry.h
    include/DemandPbrtScene/SceneProxy.h
    include/DemandPbrtScene/SceneStatistics.h
    include/DemandPbrtScene/SceneSyncState.h
    include/DemandPbrtScene/Statistics.h
    include/DemandPbrtScene/Stopwatch.h
    include/DemandPbrtScene/Timer.h
    include/DemandPbrtScene/UserInterface.h
    include/DemandPbrtScene/UserInterfaceStatistics.h
)
source_group("CMake Template Files" REGULAR_EXPRESSION "^.*\.in$")
set_target_properties(DemandPbrtSceneImpl PROPERTIES FOLDER "Examples/DemandLoading")
target_include_directories(DemandPbrtSceneImpl PUBLIC include)
target_include_directories(DemandPbrtSceneImpl PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)
target_link_libraries(DemandPbrtSceneImpl PUBLIC 
    PbrtSceneLoader
    DemandPbrtSceneKernel
    OptiXToolkit::DemandGeometry
    OptiXToolkit::DemandLoading
    OptiXToolkit::DemandMaterial
    OptiXToolkit::Error
    OptiXToolkit::Gui
    OptiXToolkit::Memory
    OptiXToolkit::OptiXMemory
    OptiXToolkit::ShaderUtil
    OptiXToolkit::Util
    OptiX::OptiX
)

add_executable(DemandPbrtScene main.cpp ReadMe.md)
target_link_libraries(DemandPbrtScene PUBLIC DemandPbrtSceneImpl)
set_target_properties(DemandPbrtScene PROPERTIES FOLDER "Examples/DemandLoading")

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
