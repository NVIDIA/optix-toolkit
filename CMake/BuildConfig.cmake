# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

#########################################################
# Build configuration

# Set the default build type to RelWithDebInfo.
set(CMAKE_BUILD_TYPE_INIT RelWithDebInfo)

# Use the shared CUDA runtime library.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# optixTexFootprint2D is hardware-accelerated in sm60+
# CUDA SDK 12.9 warns that < sm75 will be deprecated in a future release.
set(CMAKE_CUDA_ARCHITECTURES "75-virtual")

# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Use fast math for CUDA code.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# Compile options for OTK targets.
function(otk_target_options target_name)
  # Skip interface libraries, since we can't set private properties on them.
  get_target_property(target_type ${target_name} TYPE)
  if(target_type STREQUAL INTERFACE_LIBRARY)
    return()
  endif()
  set_target_properties(${target_name} PROPERTIES COMPILE_WARNING_AS_ERROR ${OTK_WARNINGS_AS_ERRORS})
  if(MSVC)
    target_compile_definitions(${target_name} PRIVATE -DNOMINMAX)
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
    # Disable warnings:
    # /wd4267 conversion from size_t
    # /wd4505 unreferenced local function has been removed
    # /wd4324 structure was padded due to alignment specifier
    # /wd4996 This function or variable may be unsafe.
    # /wd4305 truncation from 'double' to 'float'
    # /wd4245 conversion from [enum type] to 'unsigned int', signed/unsigned mismatch
    # /wd4201 nonstandard extension used: nameless struct/union
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/wd4267 /wd4505 /wd4324 /wd4996 /wd4305 /wd4245 /wd4201>
      $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options /wd4324,/wd4505>
    )
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
  endif()
endfunction(otk_target_options)

function(otk_add_library target_name)
  add_library(${target_name} ${ARGN})
  otk_target_options(${target_name})
endfunction(otk_add_library)

function(otk_add_executable target_name)
  add_executable(${target_name} ${ARGN})
  otk_target_options(${target_name})
endfunction(otk_add_executable)

