# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)
include(BuildConfig)
include(FetchGlfw)
include(FetchGlad)
include(embed_cuda)

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 11.1 REQUIRED )
endif()

embed_cuda(
  CONST HEADER UdimTextureViewerKernelCuda.h
  OUTPUT_TARGET
    udimTextureViewerKernel
  FOLDER
    Examples/DemandLoading
  LIBRARIES
    OptiXToolkit::DemandLoading 
    OptiXToolkit::OTKAppBase
    OptiXToolkit::ShaderUtil
  SOURCES
    UdimTextureViewer.cu
)

otk_add_executable( udimTextureViewer
  UdimTextureViewer.cpp
  )

target_link_libraries( udimTextureViewer
  udimTextureViewerKernel
  OptiXToolkit::DemandLoading
  OptiXToolkit::OTKAppBase
  OptiXToolkit::Gui
  OptiXToolkit::Util
  CUDA::cudart
  )

set_target_properties( udimTextureViewer PROPERTIES
  FOLDER Examples/DemandLoading
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:udimTextureViewer> )
  add_custom_command( TARGET udimTextureViewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:udimTextureViewer> $<TARGET_FILE_DIR:udimTextureViewer>
    COMMAND_EXPAND_LISTS )
endif()
