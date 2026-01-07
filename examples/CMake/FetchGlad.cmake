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
    # glad's upstream CMakeLists.txt uses an older cmake_minimum_required, which
    # fails under CMake 4.2+. We'll populate, patch that line, then add_subdirectory().
    cmake_policy(PUSH)
    if(POLICY CMP0169)
      cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Populate(glad)
    cmake_policy(POP)

    # Patch glad's cmake_minimum_required to be compatible with CMake 4.2+
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
      set(_glad_cmakelists "${glad_SOURCE_DIR}/CMakeLists.txt")
      if(EXISTS "${_glad_cmakelists}")
        file(READ "${_glad_cmakelists}" _glad_cmake_contents)
        string(REGEX REPLACE
          "([Cc][Mm][Aa][Kk][Ee]_[Mm][Ii][Nn][Ii][Mm][Uu][Mm]_[Rr][Ee][Qq][Uu][Ii][Rr][Ee][Dd]\\(VERSION)[ \t]+([0-9]+\\.[0-9]+)(\\))"
          "\\1 3.5...3.30\\3"
          _glad_cmake_contents
          "${_glad_cmake_contents}"
        )
        file(WRITE "${_glad_cmakelists}" "${_glad_cmake_contents}")
        unset(_glad_cmake_contents)
      endif()
      unset(_glad_cmakelists)
    endif()

    set( GLAD_INSTALL OFF CACHE BOOL "Generate glad installation target" )
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
