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

add_executable( testDemandLoading
  main.cpp
  DemandLoaderTestKernels.cu
  DemandLoaderTestKernels.h
  PagingSystemTestKernels.cu
  PagingSystemTestKernels.h
  TestBulkMemory.cpp
  TestBulkPinnedItemPool.cpp
  TestCheckerBoardImage.cpp
  TestDemandLoader.cpp
  TestDemandTexture.cpp
  TestDenseTexture.cpp
  TestDeviceContextImpl.cpp
  TestDeviceContextPool.cpp
  TestEXRReader.cpp
  TestEventPool.cpp
  TestFixedPool.cpp
  TestItemPool.cpp
  TestMutexArray.cpp
  TestPageTableManager.cpp
  TestPagingSystem.cpp
  TestPagingSystemKernels.cpp
  TestPinnedItemPool.cpp
  TestSparseTexture.cpp
  TestSparseTexture.cu
  TestSparseTexture.h
  TestSparseTextureWrap.cpp
  TestTextureFill.cpp
  TestTextureInstantiation.cpp
  TestTicket.cpp
  TestTileArena.cpp
  TestTileIndexing.cpp
  TestTilePool.cpp
  TestTraceFile.cpp
  )  

target_include_directories( testDemandLoading PUBLIC
  ../src
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>  # for SourceDir.h
  )

# Embed CMAKE_CURRENT_SOURCE_DIR in getSourceDir()
configure_file( SourceDir.h.in include/SourceDir.h @ONLY )

target_link_libraries( testDemandLoading
  DemandLoading
  OpenEXR::OpenEXR # for half
  gtest
  )

if( OTK_USE_CUDA_MEMORY_POOLS )
  target_compile_options( testDemandLoading PRIVATE "-DOTK_USE_CUDA_MEMORY_POOLS" )
endif()

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:testDemandLoading> )
  add_custom_command( TARGET testDemandLoading POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:testDemandLoading> $<TARGET_FILE_DIR:testDemandLoading>
    COMMAND_EXPAND_LISTS )
endif()

set_property(TARGET testDemandLoading PROPERTY FOLDER DemandLoading/tests)

# Register test cases with CTest.
gtest_discover_tests(testDemandLoading)

# The texture footprint test employs an OptiX kernel, which is compiled from CUDA to PTX.
include( embed_ptx )
embed_ptx(
  CONST HEADER TextureFootprintPTX.h
  OUTPUT_TARGET
    TextureFootprint
  PTX_LINK_LIBRARIES
    DemandLoading
  SOURCES
    TestTextureFootprint.cu
)

add_executable( testTextureFootprint
  main.cpp
  TestTextureFootprint.cpp
  TestTextureFootprint.h
  )

target_include_directories( testTextureFootprint PUBLIC
  ../src
  )

target_link_libraries( testTextureFootprint
  TextureFootprint
  DemandLoading
  gtest
  )

add_dependencies( testTextureFootprint TextureFootprint )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:testTextureFootprint> )
  add_custom_command( TARGET testTextureFootprint POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:testTextureFootprint> $<TARGET_FILE_DIR:testTextureFootprint>
    COMMAND_EXPAND_LISTS )
endif()

set_property(TARGET testTextureFootprint PROPERTY FOLDER DemandLoading/tests)

# Register test cases with CTest.
gtest_discover_tests(testTextureFootprint)
