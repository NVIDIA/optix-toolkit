# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 10.0 REQUIRED )
endif()

include(BuildConfig)
include(CTest)
include(FetchOpenEXR)
include(FetchOptiX)
include(FetchStbImage)

configure_file(src/BinaryDataDir.h.in BinaryDataDir.h)

otk_add_library(Util STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/BinaryDataDir.h
  include/OptiXToolkit/Util/AssetLocator.h
  include/OptiXToolkit/Util/EXRInputFile.h
  include/OptiXToolkit/Util/Fill.h
  include/OptiXToolkit/Util/ImageBuffer.h
  include/OptiXToolkit/Util/Logger.h
  src/AssetLocator.cpp
  src/BinaryDataDir.h.in
  src/EXRInputFile.cpp
  src/ImageBuffer.cpp
  src/Logger.cpp
  src/BinaryDataDir.h.in
  )

add_library(OptiXToolkit::Util ALIAS Util)

target_include_directories(Util PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OptiX_INCLUDE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
target_include_directories(Util PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(Util PRIVATE
  OptiX::OptiX
  CUDA::cuda_driver
  CUDA::cudart
  OpenEXR::OpenEXR
  Stb::Image
  )
target_link_libraries(Util PUBLIC OptiXToolkit::Memory)

set_target_properties(Util PROPERTIES FOLDER Examples/Libraries)

if( BUILD_TESTING )
  add_subdirectory( tests )
endif()
