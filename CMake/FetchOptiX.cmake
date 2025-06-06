# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

if (TARGET OptiX::OptiX)
  return()
endif()

find_package(CUDAToolkit REQUIRED)

# OptiX_INSTALL_DIR can be set to root of an installed OptiX SDK.
# If not specified, FetchContent is used to download the OptiX headers.
set( OptiX_INSTALL_DIR "OptiX_INSTALL_DIR-NOTFOUND" CACHE PATH "Path to OptiX installed location (optional)." )

# We use the oldest possible version of OptiX to support the widest range of driver versions.  
# Most of OTK requires OptiX 7.3, but DemandGeometry examples require 7.5 and OmmBaking requires 7.6.
include(FetchContent)
FetchContent_Declare(
    OptiX
    GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
    GIT_TAG v7.6.0
    FIND_PACKAGE_ARGS  # Try find_package first (without version constraint)
  )
FetchContent_MakeAvailable(OptiX)

# find_package sets OptiX_ROOT_DIR, while FetchContent sets optix_SOURCE_DIR.
if(OptiX_ROOT_DIR)
  set(OptiX_INCLUDE_DIR ${OptiX_ROOT_DIR}/include)
elseif(optix_SOURCE_DIR)
  set(OptiX_INCLUDE_DIR ${optix_SOURCE_DIR}/include)
else()
  message(FATAL "Problem fetching OptiX.")
endif()
message(STATUS "OptiX_INCLUDE_DIR = ${OptiX_INCLUDE_DIR}")

# Check minimum OptiX version (7.3.0 = 70300)
file(READ "${OptiX_INCLUDE_DIR}/optix.h" OPTIX_HEADER_CONTENTS)
string(REGEX MATCH "#define OPTIX_VERSION[ \t]+([0-9]+)" _ "${OPTIX_HEADER_CONTENTS}")
set(OPTIX_VERSION ${CMAKE_MATCH_1})
if(OPTIX_VERSION LESS 70300)
  message(FATAL_ERROR "OptiX version ${OPTIX_VERSION} is too old. Minimum required version is 7.3.0.")
endif()
message(STATUS "OptiX version: ${OPTIX_VERSION}")

# Header-only library
add_library(OptiX::OptiX INTERFACE IMPORTED)
target_include_directories(OptiX::OptiX INTERFACE ${OptiX_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(OptiX::OptiX INTERFACE ${CMAKE_DL_LIBS})
