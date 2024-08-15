# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
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
