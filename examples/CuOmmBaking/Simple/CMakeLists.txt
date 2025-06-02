# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)
include(BuildConfig)
include(SetCxxStandard)

set_cxx_standard( 17 )

otk_add_executable( ommBakingSimple
  simple.cpp
  simple.cu
  SourceDir.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/SourceDir.h
  )

target_link_libraries( ommBakingSimple
  OptiXToolkit::CuOmmBaking
  OptiXToolkit::Util
  OptiX::OptiX
  )

target_include_directories(ommBakingSimple PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>  # for SourceDir.h
  )

# Embed CMAKE_CURRENT_SOURCE_DIR in getSourceDir()
configure_file( SourceDir.h.in include/SourceDir.h @ONLY )

set_target_properties( ommBakingSimple PROPERTIES INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit FOLDER Examples/CmOmmBaking )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:ommBakingSimple> )
  add_custom_command( TARGET ommBakingSimple POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:ommBakingSimple> $<TARGET_FILE_DIR:ommBakingSimple>
    COMMAND_EXPAND_LISTS )
endif()
