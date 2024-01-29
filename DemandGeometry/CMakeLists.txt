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
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,\
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

include(BuildConfig)
find_package( CUDAToolkit 11.1 REQUIRED )

otk_add_library( DemandGeometry STATIC
    include/OptiXToolkit/DemandGeometry/DemandGeometry.h
    include/OptiXToolkit/DemandGeometry/GeometryLoader.h
    include/OptiXToolkit/DemandGeometry/intersectAabb.h
    include/OptiXToolkit/DemandGeometry/ProxyInstances.h
    include/OptiXToolkit/DemandGeometry/ProxyInstancesImpl.h
    src/ProxyInstances.cpp
    README.md
)
target_include_directories( DemandGeometry PUBLIC include )
target_link_libraries( DemandGeometry PUBLIC DemandLoading OptiXToolkit::Error CUDA::cuda_driver )
set_target_properties( DemandGeometry PROPERTIES FOLDER DemandLoading )

if(NOT PROJECT_IS_TOP_LEVEL)
  add_library(OptiXToolkit::DemandGeometry ALIAS DemandGeometry)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
