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

# Single CTest entry per binary: this binary creates a CUDA/OptiX context that
# is reused across cases, so one process pays context init once, not per case.
add_test( NAME testMemory COMMAND testMemory )
set_tests_properties( testMemory PROPERTIES LABELS Memory )

otk_add_executable( testError
  TestError.cpp
)
target_link_libraries(testError
  Error
  CUDA::cudart
  GTest::gtest_main
)
set_property(TARGET testError PROPERTY FOLDER Memory/Tests)
add_test( NAME testError COMMAND testError )
set_tests_properties( testError PROPERTIES LABELS Memory )

otk_add_executable(testOptiXMemory
  TestBuilders.cpp
  TestSyncRecordHeader.cpp
  OptiXFunctionTable.cpp
)
target_link_libraries(testOptiXMemory
  OptiXMemory
  Error
  CUDA::cudart
  GTest::gtest_main
)
set_property(TARGET testOptiXMemory PROPERTY FOLDER Memory/Tests)
add_test( NAME testOptiXMemory COMMAND testOptiXMemory )
set_tests_properties( testOptiXMemory PROPERTIES LABELS Memory )
