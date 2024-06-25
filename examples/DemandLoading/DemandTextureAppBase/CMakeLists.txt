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
include(FetchImgui)

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 11.1 REQUIRED )
endif()

option( BUILD_SHARED_LIBS "Build using shared libraries" ON )

otk_add_library(DemandTextureAppBase STATIC
  src/DemandTextureApp.cpp
  src/DemandTextureApp3D.cpp
  src/ShapeMaker.cpp
  include/OptiXToolkit/DemandTextureAppBase/SimpleBsdf.h
  include/OptiXToolkit/DemandTextureAppBase/DemandTextureApp.h
  include/OptiXToolkit/DemandTextureAppBase/DemandTextureApp3D.h
  include/OptiXToolkit/DemandTextureAppBase/DemandTextureAppDeviceUtil.h
  include/OptiXToolkit/DemandTextureAppBase/LaunchParams.h
  include/OptiXToolkit/DemandTextureAppBase/PerDeviceOptixState.h
  include/OptiXToolkit/DemandTextureAppBase/ShapeMaker.h
  )

add_library(OptiXToolkit::DemandTextureAppBase ALIAS DemandTextureAppBase)

target_include_directories(DemandTextureAppBase PUBLIC include)

target_link_libraries(DemandTextureAppBase
  PUBLIC
  OptiXToolkit::DemandLoading
  OptiXToolkit::Gui
  OptiXToolkit::ImageSources
  OptiXToolkit::ShaderUtil
  OptiXToolkit::Util
  imgui::imgui
  CUDA::cudart
  )

set_target_properties(DemandTextureAppBase PROPERTIES FOLDER Examples/DemandLoading)
