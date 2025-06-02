# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
include(Policies)
include(SetCxxStandard)

project(DemandLoading LANGUAGES C CXX CUDA)
set_cxx_standard(11)
set(CMAKE_CXX_EXTENSIONS OFF)
include(GNUInstallDirs)
include(BuildConfig)
include(CTest)

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 11.1 REQUIRED )
endif()

if ( CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.2 )
  option( OTK_USE_CUDA_MEMORY_POOLS "Use CUDA memory pools and stream-ordered allocators" ON )
else()
  option( OTK_USE_CUDA_MEMORY_POOLS "Use CUDA memory pools and stream-ordered allocators" OFF )
endif()

if(NOT TARGET OptiX::OptiX)
  # Set OptiX_INSTALL_DIR to the root of the OptiX SDK when configuring CMake.
  set(OptiX_INSTALL_DIR "OptiX_INSTALL_DIR-NOTFOUND" CACHE PATH "Path to OptiX installed location.")
  find_package(OptiX 7.3 REQUIRED)
endif()

option( BUILD_SHARED_LIBS "Build using shared libraries" ON )

add_subdirectory(DemandGeometry)
add_subdirectory(DemandLoading)
add_subdirectory(DemandMaterial)
add_subdirectory(ImageSource)
