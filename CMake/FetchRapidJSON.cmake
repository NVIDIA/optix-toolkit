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
FetchContent_MakeAvailable(RapidJSON)
# Let find_package know we have it
set(RapidJSON_FOUND ON PARENT_SCOPE)


