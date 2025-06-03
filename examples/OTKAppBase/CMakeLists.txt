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

otk_add_library(OTKAppBase STATIC
  src/OTKApp.cpp
#  src/OTKApp3D.cpp
  src/OTKAppShapeMaker.cpp
  include/OptiXToolkit/OTKAppBase/OTKAppSimpleBsdf.h
  include/OptiXToolkit/OTKAppBase/OTKApp.h
#  include/OptiXToolkit/OTKAppBase/OTKApp3D.h
  include/OptiXToolkit/OTKAppBase/OTKAppDeviceUtil.h
  include/OptiXToolkit/OTKAppBase/OTKAppLaunchParams.h
  include/OptiXToolkit/OTKAppBase/OTKAppOptixPrograms.h
  include/OptiXToolkit/OTKAppBase/OTKAppPerDeviceOptixState.h
  include/OptiXToolkit/OTKAppBase/OTKAppShapeMaker.h
  )

add_library(OptiXToolkit::OTKAppBase ALIAS OTKAppBase)

target_include_directories(OTKAppBase PUBLIC include)

target_link_libraries(OTKAppBase
  PUBLIC
  OptiXToolkit::DemandLoading
  OptiXToolkit::Gui
  OptiXToolkit::ImageSources
  OptiXToolkit::OptiXMemory
  OptiXToolkit::ShaderUtil
  OptiXToolkit::Util
  imgui::imgui
  CUDA::cudart
  )

set_target_properties(OTKAppBase PROPERTIES FOLDER Examples/DemandLoading)
