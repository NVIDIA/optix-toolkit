# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

if( TARGET glfw )
    return()
endif()

if(OTK_USE_VCPKG)
    find_package(glfw3 CONFIG REQUIRED)
    return()
endif()

function(glfw_folders)
  foreach(_target glfw update_mappings)
    if(TARGET ${_target})
      set_property(TARGET ${_target} PROPERTY FOLDER ThirdParty/GLFW3)
    endif()
  endforeach()
endfunction()

if( NOT OTK_FETCH_CONTENT )
    find_package( glfw3 REQUIRED )
    glfw_folders()
    return()
endif()

set( GLFW_BUILD_DOCS OFF CACHE BOOL "Build the GLFW documentation" )
set( GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs" )
set( GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs" )
set( GLFW_INSTALL OFF CACHE BOOL "Generate GLFW installation target")

include(FetchContent)

message(VERBOSE "Finding glfw3...")
FetchContent_Declare(
    glfw3
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.7
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS
    )
FetchContent_MakeAvailable(glfw3)
# Let find_package know we have it
set(glfw3_FOUND ON PARENT_SCOPE)

glfw_folders()
