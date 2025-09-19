# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)
include(BuildConfig)
include(FetchGlfw)
include(FetchGlad)
include(embed_cuda)

embed_cuda(
  CONST HEADER TexturePaintingKernelCuda.h
  OUTPUT_TARGET
    texturePaintingKernel
  FOLDER
    Examples/DemandLoading
  LIBRARIES
    OptiXToolkit::DemandLoading 
    OptiXToolkit::OTKAppBase
    OptiXToolkit::ShaderUtil
  SOURCES
    TexturePainting.cu
)

otk_add_executable( texturePainting
  CanvasImage.cpp
  CanvasImage.h
  TexturePainting.cpp
  )

target_link_libraries( texturePainting
  texturePaintingKernel
  OptiXToolkit::DemandLoading
  OptiXToolkit::OTKAppBase
  OptiXToolkit::Gui
  OptiXToolkit::Util
  )

set_target_properties( texturePainting PROPERTIES
  FOLDER Examples/DemandLoading
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:texturePainting> )
  add_custom_command( TARGET texturePainting POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:texturePainting> $<TARGET_FILE_DIR:texturePainting>
    COMMAND_EXPAND_LISTS )
endif()
