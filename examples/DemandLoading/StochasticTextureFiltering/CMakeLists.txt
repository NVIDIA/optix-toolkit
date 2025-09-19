# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)
include(BuildConfig)
include(FetchGlfw)
include(FetchGlad)
include(embed_cuda)

embed_cuda(
  CONST HEADER StochasticTextureFilteringKernel.h
  OUTPUT_TARGET
    stochasticTextureFilteringKernel
  FOLDER
    Examples/DemandLoading
  LIBRARIES
    OptiXToolkit::DemandLoading 
    OptiXToolkit::OTKAppBase
    OptiXToolkit::ShaderUtil
  SOURCES
    StochasticTextureFiltering.cu
)

otk_add_executable( stochasticTextureFiltering
  StochasticTextureFiltering.cpp
  )

target_link_libraries( stochasticTextureFiltering
  PUBLIC
  stochasticTextureFilteringKernel
  OptiXToolkit::DemandLoading
  OptiXToolkit::OTKAppBase
  OptiXToolkit::Gui
  OptiXToolkit::Util
  )

set_target_properties( stochasticTextureFiltering PROPERTIES
  FOLDER Examples/DemandLoading
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:stochasticTextureFiltering> )
  add_custom_command( TARGET stochasticTextureFiltering POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:stochasticTextureFiltering> $<TARGET_FILE_DIR:stochasticTextureFiltering>
    COMMAND_EXPAND_LISTS )
endif()
