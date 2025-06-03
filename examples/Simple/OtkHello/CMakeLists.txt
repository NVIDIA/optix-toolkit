# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)
include(BuildConfig)
include(FetchGlad)
include(embed_cuda)

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 11.0 REQUIRED )
endif()

embed_cuda(
  GEN_DEBUG
  CONST HEADER otkHelloKernelCuda.h
  OUTPUT_TARGET
    otkHelloKernel
  FOLDER
    Examples/Simple
  LIBRARIES
    OptiXToolkit::ShaderUtil
    OptiX::OptiX
  SOURCES
    draw_solid_color.cu
)

otk_add_executable( otkHello
  hello.cpp
  hello.h
  )

target_link_libraries( otkHello
  otkHelloKernel
  OptiXToolkit::Gui
  OptiXToolkit::OptiXMemory
  OptiXToolkit::Util
  OptiX::OptiX
  CUDA::cuda_driver
  CUDA::cudart
  )

set_target_properties( otkHello PROPERTIES
  FOLDER Examples/Simple
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:otkHello> )
  add_custom_command( TARGET otkHello POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:otkHello> $<TARGET_FILE_DIR:otkHello>
    COMMAND_EXPAND_LISTS )
endif()
