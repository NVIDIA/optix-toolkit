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

include(GNUInstallDirs)
include(BuildConfig)
include(FetchGlfw)
include(FetchGlad)
include(embed_cuda)

embed_cuda(
  CONST HEADER RayConesKernelCuda.h
  OUTPUT_TARGET
    rayConesKernel
  FOLDER
    Examples/DemandLoading
  LIBRARIES
    OptiXToolkit::DemandLoading 
    OptiXToolkit::DemandTextureAppBase
    OptiXToolkit::ShaderUtil
  SOURCES
    RayCones.cu
)

otk_add_executable( rayCones
  RayCones.cpp
  )

target_link_libraries( rayCones
  PUBLIC
  rayConesKernel
  OptiXToolkit::DemandLoading
  OptiXToolkit::DemandTextureAppBase
  OptiXToolkit::Gui
  OptiXToolkit::ImageSources
  OptiXToolkit::Util
  )

set_target_properties( rayCones PROPERTIES
  FOLDER Examples/DemandLoading
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:rayCones> )
  add_custom_command( TARGET rayCones POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:rayCones> $<TARGET_FILE_DIR:rayCones>
    COMMAND_EXPAND_LISTS )
endif()
