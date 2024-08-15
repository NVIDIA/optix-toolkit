# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

if( TARGET glad::glad )
    return()
endif()

if(OTK_USE_VCPKG)
    find_package(glad CONFIG REQUIRED)
    return()
endif()

function(glad_folders)
  foreach(_target glad glad-generate-files)
    if(TARGET ${_target})
      set_property(TARGET ${_target} PROPERTY FOLDER ThirdParty/glad)
    endif()
  endforeach()
endfunction()

if( NOT OTK_FETCH_CONTENT )
  find_package( glad REQUIRED )
  glad_folders()
  return()
endif()

include(FetchContent)

set( GLAD_INSTALL OFF CACHE BOOL "Generate glad installation target" )

message(VERBOSE "Finding glad...")
FetchContent_Declare(
  glad
  GIT_REPOSITORY https://github.com/Dav1dde/glad
  GIT_TAG v0.1.36
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS
)
FetchContent_GetProperties(glad)
if(NOT glad_POPULATED)
    FetchContent_Populate(glad)
    set(GLAD_PROFILE "core" CACHE STRING "OpenGL profile")
    set(GLAD_API "gl=" CACHE STRING "API type/version pairs, like \"gl=3.2,gles=\", no version means latest")
    set(GLAD_GENERATOR "c" CACHE STRING "Language to generate the binding for")
    add_subdirectory(${glad_SOURCE_DIR} ${glad_BINARY_DIR})
endif()
# Let find_package know we have it
set(glad_FOUND ON PARENT_SCOPE)

set_target_properties(glad PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (NOT TARGET glad::glad)
    add_library(glad::glad ALIAS glad)
endif()

glad_folders()
