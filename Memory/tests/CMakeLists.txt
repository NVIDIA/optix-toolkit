# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include( FetchGtest )
include( GoogleTest )

otk_add_executable( testMemory
  TestAllocators.cpp
  TestBinnedSuballocator.cpp
  TestDeviceBuffer.cpp
  TestDeviceMemoryPools.cpp
  TestDeviceMemoryPools.cu
  TestDeviceMemoryPools.h
  TestFixedSuballocator.cpp
  TestHeapSuballocator.cpp
  TestMemoryPool.cpp
  TestRingSuballocator.cpp
  TestSyncVector.cpp
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
  FOLDER Memory/Tests
)

# Register test cases with CTest.
gtest_discover_tests(testMemory PROPERTIES LABELS Memory)

otk_add_executable( testError
  TestError.cpp
)
target_link_libraries(testError
  Error
  CUDA::cudart
  GTest::gtest_main
)
set_property(TARGET testError PROPERTY FOLDER Memory/Tests)
gtest_discover_tests(testError PROPERTIES LABELS Memory)

otk_add_executable(testOptiXMemory
  TestBuilders.cpp
  TestSyncRecordHeader.cpp
)
target_link_libraries(testOptiXMemory
  OptiXMemory
  Error
  CUDA::cudart
  GTest::gtest_main
)
set_property(TARGET testOptiXMemory PROPERTY FOLDER Memory/Tests)
gtest_discover_tests(testOptiXMemory PROPERTIES LABELS Memory)
