#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
