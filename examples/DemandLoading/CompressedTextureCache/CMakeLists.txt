# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

set_cxx_standard( 17 )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

otk_add_executable( compressedTextureCache
  CompressedTextureCache.cpp
  )

target_link_libraries( compressedTextureCache
  OptiXToolkit::DemandLoading
  )

set_target_properties( compressedTextureCache PROPERTIES
  FOLDER Examples/DemandLoading
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )
