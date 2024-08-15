# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)
include(BuildConfig)
include(FetchGlfw)
include(FetchGlad)
include(embed_cuda)

embed_cuda(
  HEADER TextureVariantViewerCuda.h
  OUTPUT_TARGET
    TextureVariantViewerKernel
  FOLDER
    Examples/DemandLoading
  LIBRARIES
    OptiXToolkit::DemandLoading 
    OptiXToolkit::DemandTextureAppBase
    OptiXToolkit::ShaderUtil
  SOURCES
    TextureVariantViewer.cu
)

otk_add_executable( textureVariantViewer
  TextureVariantViewer.cpp
  )

target_link_libraries( textureVariantViewer
  TextureVariantViewerKernel
  OptiXToolkit::DemandLoading
  OptiXToolkit::DemandTextureAppBase
  OptiXToolkit::Gui
  OptiXToolkit::ImageSources
  OptiXToolkit::Util
  glad::glad
  glfw
  )

set_target_properties( textureVariantViewer PROPERTIES
  FOLDER Examples/DemandLoading
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:textureVariantViewer> )
  add_custom_command( TARGET textureVariantViewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:textureVariantViewer> $<TARGET_FILE_DIR:textureVariantViewer>
    COMMAND_EXPAND_LISTS )
endif()
