# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)
include(BuildConfig)
include(FetchGlad)
include(embed_cuda)

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 11.1 REQUIRED )
endif()

embed_cuda(
  CONST HEADER TextureKernelCuda.h
  OUTPUT_TARGET
    demandLoadTextureKernel
  FOLDER
    Examples/DemandLoading
  LIBRARIES
    OptiXToolkit::DemandLoading 
    OptiXToolkit::ShaderUtil
  SOURCES
    textureKernel.cu
)

otk_add_executable( demandLoadTexture
  texture.cpp
  textureKernel.h
  SourceDir.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/SourceDir.h
  )

target_link_libraries( demandLoadTexture
  demandLoadTextureKernel
  OptiXToolkit::DemandLoading
  OptiXToolkit::Gui
  OptiXToolkit::ImageSources
  OptiXToolkit::OptiXMemory
  OptiXToolkit::Util
  CUDA::cudart
  )

target_include_directories( demandLoadTexture PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>  # for SourceDir.h
  )

# Embed CMAKE_CURRENT_SOURCE_DIR in getSourceDir()
configure_file( SourceDir.h.in include/SourceDir.h @ONLY )

set_target_properties( demandLoadTexture PROPERTIES
  FOLDER Examples/DemandLoading
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:demandLoadTexture> )
  add_custom_command( TARGET demandLoadTexture POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:demandLoadTexture> $<TARGET_FILE_DIR:demandLoadTexture>
    COMMAND_EXPAND_LISTS )
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
