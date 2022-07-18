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

include( ../../CMake/FetchGtest.cmake )

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
  ${CMAKE_CURRENT_BINARY_DIR}  # for Config.h
  ../src
  )

target_link_libraries( testDemandLoading
  DemandLoading
  OtkUtil
  OpenEXR::OpenEXR # for half
  gtest
  )

# The texture footprint test employs an OptiX kernel, which is compiled from CUDA to PTX.
include(../../CMake/embed_ptx.cmake)
embed_ptx(
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
  ${CMAKE_CURRENT_BINARY_DIR}  # for Config.h
  ../src
  )

target_link_libraries( testTextureFootprint
  TextureFootprint
  DemandLoading
  gtest
  )

add_dependencies( testTextureFootprint TextureFootprint )

get_target_property( _bin testDemandLoading RUNTIME_OUTPUT_DIRECTORY )
file( INSTALL Textures DESTINATION ${_bin} )
