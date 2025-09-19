# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
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
    OptiXToolkit::OTKAppBase
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
  OptiXToolkit::OTKAppBase
  OptiXToolkit::Gui
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
