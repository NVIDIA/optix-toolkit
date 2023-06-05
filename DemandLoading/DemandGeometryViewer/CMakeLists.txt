#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

include(BuildConfig)
include(embed_ptx)

find_package(OptiX 7.5)
if(NOT OptiX_FOUND)
  message(WARNING "Cannot build DemandGeometryViewer: it requires OptiX 7.5 or later")
  return()
endif()

find_package( CUDAToolkit 11.1 REQUIRED )

embed_ptx(
    CONST HEADER DemandGeometryViewerKernelPTX.h
    OUTPUT_TARGET DemandGeometryViewerKernel
    PTX_INCLUDE_DIRECTORIES include
    PTX_LINK_LIBRARIES
        OptiXToolkit::DemandGeometry
        OptiXToolkit::DemandLoading
        OptiXToolkit::ShaderUtil
    SOURCES
        DemandGeometryViewer.cu
        DemandMaterial.cu
        Sphere.cu
    FOLDER Examples/DemandLoading
)

otk_add_executable( DemandGeometryViewer
    DemandGeometryViewer.h
    DemandGeometryViewer.cpp
    DemandMaterial.h
    DemandMaterial.cpp
    SphereInstances.h
    SphereInstances.cpp
)
target_link_libraries( DemandGeometryViewer PUBLIC
    DemandGeometryViewerKernel
    DemandGeometry
    ShaderUtil
    Gui
    Util
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
