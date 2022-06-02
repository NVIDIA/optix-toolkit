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

add_executable( testDemandLoading
  main.cpp
  DemandLoaderTestKernels.cu
  DemandLoaderTestKernels.h
  PagingSystemTestKernels.cu
  PagingSystemTestKernels.h
  TestBulkMemory.cpp
  TestBulkPinnedItemPool.cpp
  TestDemandLoader.cpp
  TestDemandTexture.cpp
  TestDenseTexture.cpp
  TestDeviceContextImpl.cpp
  TestDeviceContextPool.cpp
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

configure_file( Config.h.in Config.h @ONLY )

target_include_directories( testDemandLoading PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}  # for Config.h
  ${CMAKE_SOURCE_DIR}/lib/DemandLoading
  )

target_link_libraries( testDemandLoading
  DemandLoading
  gtest
  )

set_property( TARGET testDemandLoading PROPERTY CUDA_ARCHITECTURES "60" )

# The texture footprint test employs an OptiX kernel, which is compiled from CUDA to PTX.
add_library( TextureFootprint OBJECT TestTextureFootprint.cu )
set_property( TARGET TextureFootprint PROPERTY CUDA_PTX_COMPILATION ON )
set_property( TARGET TextureFootprint PROPERTY CUDA_ARCHITECTURES "60" )
target_include_directories( TextureFootprint PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${OptiX_INCLUDE}
  )

add_executable( testTextureFootprint
  main.cpp
  TestTextureFootprint.cpp
  TestTextureFootprint.h
  )

target_include_directories( testTextureFootprint PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}  # for Config.h
  ${CMAKE_SOURCE_DIR}/lib/DemandLoading
  )

target_link_libraries( testTextureFootprint
  DemandLoading
  gtest
  )

set_property( TARGET testTextureFootprint PROPERTY CUDA_ARCHITECTURES "60" )
