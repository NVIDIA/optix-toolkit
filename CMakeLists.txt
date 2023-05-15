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

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
include(Policies)

project(Memory LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

include(BuildConfig)
include(CTest)

find_package( CUDAToolkit 11.1 REQUIRED )

if ( CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.2 )
  option( OTK_USE_CUDA_MEMORY_POOLS "Use CUDA memory pools and stream-ordered allocators" ON )
else()
  option( OTK_USE_CUDA_MEMORY_POOLS "Use CUDA memory pools and stream-ordered allocators" OFF )
endif()

add_library(Memory INTERFACE)
set_property(TARGET Memory PROPERTY FOLDER Memory)

if(NOT PROJECT_IS_TOP_LEVEL)
  add_library(OptiXToolkit::Memory ALIAS Memory)
endif()

target_sources(Memory
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/Memory/Allocators.h
  include/OptiXToolkit/Memory/Assert.h
  include/OptiXToolkit/Memory/BinnedSuballocator.h
  include/OptiXToolkit/Memory/BitCast.h
  include/OptiXToolkit/Memory/CudaCheck.h
  include/OptiXToolkit/Memory/DeviceBuffer.h
  include/OptiXToolkit/Memory/DeviceFixedPool.h
  include/OptiXToolkit/Memory/DeviceRingBuffer.h
  include/OptiXToolkit/Memory/FixedSuballocator.h
  include/OptiXToolkit/Memory/HeapSuballocator.h
  include/OptiXToolkit/Memory/MemoryBlockDesc.h
  include/OptiXToolkit/Memory/MemoryPool.h
  include/OptiXToolkit/Memory/RingSuballocator.h
)

target_include_directories( Memory INTERFACE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_link_libraries( Memory INTERFACE
  CUDA::cuda_driver
  )

if( OTK_USE_CUDA_MEMORY_POOLS )
  target_compile_options( Memory INTERFACE "-DOTK_USE_CUDA_MEMORY_POOLS" )
endif()

if( BUILD_TESTING )
  add_subdirectory( tests )
endif()
