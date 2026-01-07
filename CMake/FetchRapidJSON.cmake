# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

if(TARGET RapidJSON)
  return()
endif()

if(OTK_USE_VCPKG)
    find_package(RapidJSON CONFIG REQUIRED)
    return()
endif()

if( NOT OTK_FETCH_CONTENT )
  find_package( RapidJSON REQUIRED )
  return()
endif()

include(FetchContent)

message(VERBOSE "Finding RapidJSON...")
FetchContent_Declare(
  RapidJSON
  GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
  GIT_TAG v1.1.0
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS
)

# RapidJSON is header-only. Avoid add_subdirectory() (and thus RapidJSON's own
# CMakeLists.txt / cmake_minimum_required) by only populating the sources.
FetchContent_GetProperties(RapidJSON)
if(NOT rapidjson_POPULATED)
  # CMake 4.2 warns that calling FetchContent_Populate() after FetchContent_Declare()
  # is deprecated under policy CMP0169, but FetchContent_MakeAvailable() would
  # add_subdirectory() RapidJSON and trip its ancient cmake_minimum_required().
  cmake_policy(PUSH)
  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()
  FetchContent_Populate(RapidJSON)
  cmake_policy(POP)
endif()

# Export source dir for NeuralTextures include path logic
set(RapidJSON_SOURCE_DIR ${rapidjson_SOURCE_DIR})
set(RapidJSON_SOURCE_DIR ${rapidjson_SOURCE_DIR} PARENT_SCOPE)

# Let find_package know we have it
set(RapidJSON_FOUND ON PARENT_SCOPE)
