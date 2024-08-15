# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(BuildConfig)

otk_add_library( ImageSources STATIC
  src/DeviceConstantImage.cpp
  src/DeviceConstantImageKernels.cu
  src/DeviceMandelbrotImage.cpp
  src/DeviceMandelbrotImageKernels.cu
  src/ImageSources.cpp
  )
set_property(TARGET ImageSources PROPERTY FOLDER DemandLoading)

add_library(OptiXToolkit::ImageSources ALIAS ImageSources)

target_sources(ImageSources
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/ImageSources/DeviceConstantImage.h
  include/OptiXToolkit/ImageSources/DeviceConstantImageParams.h
  include/OptiXToolkit/ImageSources/DeviceMandelbrotImage.h
  include/OptiXToolkit/ImageSources/DeviceMandelbrotParams.h
  include/OptiXToolkit/ImageSources/ImageSources.h
  include/OptiXToolkit/ImageSources/MultiCheckerImage.h
)

target_include_directories( ImageSources
  PUBLIC
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  PRIVATE
  src
  )

target_link_libraries( ImageSources
  PRIVATE
  OptiXToolkit::ImageSource
  OptiXToolkit::Util
)

set_target_properties(ImageSources PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
