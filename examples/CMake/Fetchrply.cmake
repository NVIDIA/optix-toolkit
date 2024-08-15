# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

if( TARGET rply::rply )
    return()
endif()

if(OTK_USE_VCPKG)
    find_package(rply CONFIG REQUIRED)
    return()
endif()

if( NOT OTK_FETCH_CONTENT )
  find_package( rply REQUIRED )
  return()
endif()

include(FetchContent)

message(VERBOSE "Fetching rply...")
FetchContent_Declare(
  rply
  GIT_REPOSITORY https://github.com/diegonehab/rply.git
  GIT_TAG 75639e3f12755ca64559d45a4a0bcffc52e04ef8 # v1.1.4
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(rply)
# Let find_package know we have it
set(rply_FOUND ON PARENT_SCOPE)

configure_file(${rply_SOURCE_DIR}/rply.h ${rply_SOURCE_DIR}/include/rply/rply.h COPYONLY)
add_library(rply STATIC
    ${rply_SOURCE_DIR}/rply.h
    ${rply_SOURCE_DIR}/rply.c
    ${rply_SOURCE_DIR}/rplyfile.h
)
target_include_directories(rply PUBLIC ${rply_SOURCE_DIR}/include)
set_target_properties(rply PROPERTIES FOLDER ThirdParty)

if (NOT TARGET rply::rply)
    add_library(rply::rply ALIAS rply)
endif()
