# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)
include(BuildConfig)

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 11.1 REQUIRED )
endif()

otk_add_executable( demandLoadSimple
  simple.cpp
  Simple.h
  PageRequester.cu
  )

target_link_libraries( demandLoadSimple
  OptiXToolkit::DemandLoading
  OptiXToolkit::Util
  CUDA::cudart
  )

set_target_properties( demandLoadSimple PROPERTIES
  FOLDER Examples/DemandLoading
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:demandLoadSimple> )
  add_custom_command( TARGET demandLoadSimple POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:demandLoadSimple> $<TARGET_FILE_DIR:demandLoadSimple>
    COMMAND_EXPAND_LISTS )
endif()
