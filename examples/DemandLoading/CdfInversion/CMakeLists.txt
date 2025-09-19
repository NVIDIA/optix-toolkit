# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)
include(BuildConfig)
include(FetchGlfw)
include(FetchGlad)
include(embed_cuda)

embed_cuda(
  CONST HEADER CdfInversionKernelCuda.h
  OUTPUT_TARGET
    cdfInversionKernel
  FOLDER
    Examples/DemandLoading
  LIBRARIES
    OptiXToolkit::DemandLoading 
    OptiXToolkit::OTKAppBase
    OptiXToolkit::ShaderUtil
  SOURCES
    CdfInversion.cu
)

otk_add_executable( cdfInversion
  CdfInversion.cpp
  )

target_link_libraries( cdfInversion
  PUBLIC
  cdfInversionKernel
  OptiXToolkit::DemandLoading
  OptiXToolkit::OTKAppBase
  OptiXToolkit::Gui
  OptiXToolkit::Util
  )

set_target_properties( cdfInversion PROPERTIES
  FOLDER Examples/DemandLoading
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:cdfInversion> )
  add_custom_command( TARGET cdfInversion POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:cdfInversion> $<TARGET_FILE_DIR:cdfInversion>
    COMMAND_EXPAND_LISTS )
endif()
