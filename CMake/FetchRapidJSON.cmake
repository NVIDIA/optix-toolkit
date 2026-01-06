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

# Disable RapidJSON docs/examples/tests (tests pull in their own gtest)
set(RAPIDJSON_BUILD_DOC OFF CACHE BOOL "Build RapidJSON documentation" FORCE)
set(RAPIDJSON_BUILD_EXAMPLES OFF CACHE BOOL "Build RapidJSON examples" FORCE)
set(RAPIDJSON_BUILD_TESTS OFF CACHE BOOL "Build RapidJSON tests" FORCE)

# Standard FetchContent; options above prevent gtest conflicts
FetchContent_MakeAvailable(RapidJSON)
# Export source dir for NeuralTextures include path logic
set(RapidJSON_SOURCE_DIR ${rapidjson_SOURCE_DIR})
set(RapidJSON_SOURCE_DIR ${rapidjson_SOURCE_DIR} PARENT_SCOPE)

# Let find_package know we have it
set(RapidJSON_FOUND ON PARENT_SCOPE)


