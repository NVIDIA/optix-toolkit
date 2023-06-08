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

#########################################################
# Build configuration

# Set the default build type to RelWithDebInfo.
set(CMAKE_BUILD_TYPE_INIT RelWithDebInfo)

# optixTexFootprint2D is hardware-accelerated in sm60+
set(CMAKE_CUDA_ARCHITECTURES "60-virtual")

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
  set_target_properties(${target_name} PROPERTIES COMPILE_WARNING_AS_ERROR ON)
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

