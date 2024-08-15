# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

# (Stb is the library author's initials)
if(TARGET Stb::Image)
    return()
endif()

if(OTK_USE_VCPKG)
    find_package(Stb REQUIRED)
else()
    include(FetchContent)

    # Instead of fetching isolated files from master (which can shift implementations),
    # get the whole repository at the same commit as the vcpkg recipe for consistency.
    message(VERBOSE "Finding sbt...")
    FetchContent_Declare(
        Stb
        GIT_REPOSITORY  https://github.com/nothings/stb.git
        GIT_TAG         5736b15f7ea0ffb08dd38af21067c314d6a3aae9 # committed on 2023-04-11
        GIT_SHALLOW     OFF         # we need a deep clone to get this hash.
        CONFIGURE_COMMAND   ""      # No configure step
        BUILD_COMMAND       ""      # No build step
    )
    FetchContent_MakeAvailable(Stb)
    if(NOT stb_SOURCE_DIR)
        message(FATAL_ERROR "Could not locate stb source")
    else()
        # Let find_package know we have it
        set(stb_FOUND ON PARENT_SCOPE)
    endif()

    set(Stb_INCLUDE_DIR "${stb_SOURCE_DIR}")
endif()

# StbImage static library
#
# Build the implementation once so that every target linking against this
# header-only library only needs declarations and need not recompile the
# implementation.
#
add_library(StbImage STATIC
    ${CMAKE_CURRENT_LIST_DIR}/stb.cpp
    ${Stb_INCLUDE_DIR}/stb_image.h
    ${Stb_INCLUDE_DIR}/stb_image_write.h
)
target_include_directories(StbImage PUBLIC ${Stb_INCLUDE_DIR})
if(NOT MSVC)
  target_compile_options(StbImage PRIVATE -Wno-missing-field-initializers)
endif()
set_target_properties(StbImage PROPERTIES FOLDER ThirdParty)

add_library(Stb::Image ALIAS StbImage)
