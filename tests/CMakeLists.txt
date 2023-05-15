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

include( FetchGtest )
include( GoogleTest )

add_executable( testMemory
  TestAllocators.cpp
  TestBinnedSuballocator.cpp
  TestDeviceMemoryPools.cpp
  TestDeviceMemoryPools.cu
  TestDeviceMemoryPools.h
  TestFixedSuballocator.cpp
  TestHeapSuballocator.cpp
  TestMemoryPool.cpp
  TestRingSuballocator.cpp
  TestSyncVectorHeader.cpp
  )
target_link_libraries( testMemory
  Memory
  CUDA::cudart
  GTest::gtest_main
  )
if( OTK_USE_CUDA_MEMORY_POOLS )
  target_compile_definitions( testMemory PRIVATE OTK_USE_CUDA_MEMORY_POOLS )
endif()

set_target_properties( testMemory PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER Memory/tests
)

# Register test cases with CTest.
gtest_discover_tests(testMemory)

add_executable(testOptiXMemory
  TestBuilders.cpp
  TestSyncRecordHeader.cpp
)
target_link_libraries(testOptiXMemory
  OptiXMemory
  Error
  CUDA::cudart
  gtest_main
)
set_property(TARGET testOptiXMemory PROPERTY FOLDER Memory/tests)
gtest_discover_tests(testOptiXMemory)
