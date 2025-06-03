# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)
include(BuildConfig)
include(embed_cuda)
include(FetchGlad)
include(FetchGlfw)
include(SetCxxStandard)

set_cxx_standard( 17 )

embed_cuda(
  CONST HEADER CuOmmBakingViewerKernelCuda.h
  OUTPUT_TARGET
    ommBakingViewerKernel
  FOLDER
    Examples/CmOmmBaking
  LIBRARIES
    OptiXToolkit::CuOmmBaking 
    OptiXToolkit::ShaderUtil
  SOURCES
    CuOmmBakingViewer.cu
)

otk_add_executable( ommBakingViewer
  CuOmmBakingViewer.cpp
  CuOmmBakingApp.cu
  CuOmmBakingApp.cpp
  CuOmmBakingApp.h
  CuOmmBakingAppDeviceUtil.h
  LaunchParams.h
  PerDeviceOptixState.h
  Procedural.h
  Util/interval_math.h
  SourceDir.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/SourceDir.h
  )

source_group( "Header Files\\Util" FILES
  Util/interval_math.h
)

target_include_directories(ommBakingViewer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>  # for SourceDir.h
  )

# Embed CMAKE_CURRENT_SOURCE_DIR in getSourceDir()
configure_file( SourceDir.h.in include/SourceDir.h @ONLY )

target_link_libraries( ommBakingViewer
  PUBLIC  
  OptiXToolkit::CuOmmBaking
  OptiXToolkit::Gui
  OptiXToolkit::OptiXMemory
  OptiXToolkit::ShaderUtil
  OptiXToolkit::Util
  OptiX::OptiX
  PRIVATE
  ommBakingViewerKernel
  glfw
  glad::glad
  )

set_target_properties( ommBakingViewer PROPERTIES
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit
  FOLDER Examples/CmOmmBaking )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:ommBakingViewer> )
  add_custom_command( TARGET ommBakingViewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:ommBakingViewer> $<TARGET_FILE_DIR:ommBakingViewer>
    COMMAND_EXPAND_LISTS )
endif()
