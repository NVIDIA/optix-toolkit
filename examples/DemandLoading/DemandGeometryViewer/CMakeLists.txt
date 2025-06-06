# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(BuildConfig)
include(embed_cuda)
include(FetchOptiX)

find_package( CUDAToolkit 11.1 REQUIRED )

embed_cuda(
    CONST HEADER DemandGeometryViewerKernelCuda.h
    OUTPUT_TARGET DemandGeometryViewerKernel
    INCLUDES include
    LIBRARIES
        OptiXToolkit::DemandGeometry
        OptiXToolkit::DemandLoading
        OptiXToolkit::DemandMaterial
        OptiXToolkit::ShaderUtil
    SOURCES
        DemandGeometryViewer.cu
        Sphere.cu
    FOLDER Examples/DemandLoading
)

otk_add_executable( DemandGeometryViewer
    DemandGeometryViewer.h
    DemandGeometryViewer.cpp
    SphereInstances.h
    SphereInstances.cpp
)
target_link_libraries( DemandGeometryViewer PUBLIC
    DemandGeometryViewerKernel
    OptiXToolkit::DemandGeometry
    OptiXToolkit::DemandMaterial
    OptiXToolkit::OptiXMemory
    OptiXToolkit::ShaderUtil
    OptiXToolkit::Gui
    OptiXToolkit::Util
    OptiXToolkit::Error
    CUDA::cudart
)
set_target_properties(DemandGeometryViewer PROPERTIES FOLDER Examples/DemandLoading)
if(WIN32)
    target_compile_definitions(DemandGeometryViewer PRIVATE NOMINMAX)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
