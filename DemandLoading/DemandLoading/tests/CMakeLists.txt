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
include( embed_cuda )

embed_cuda( CONST HEADER TestDemandLoadingKernelsCuda.h
    PTX
    OUTPUT_TARGET testDemandLoadingKernels
    LIBRARIES
        DemandLoading
        ShaderUtil
    SOURCES DeferredImageLoadingKernels.cu
    FOLDER DemandLoading/Tests
)

otk_add_executable( testDemandLoading
  main.cpp
  DeferredImageLoading.cpp
  DeferredImageLoadingKernels.h
  DemandLoaderTestKernels.cu
  DemandLoaderTestKernels.h
  DeviceConstantImage.cpp
  DeviceConstantImageKernels.cu
  PagingSystemTestKernels.cu
  PagingSystemTestKernels.h
  TestContextSaver.cpp
  TestDemandLoader.cpp
  TestDemandPageLoader.cpp
  TestDemandTexture.cpp
  TestDenseTexture.cpp
  TestDeviceContextImpl.cpp
  TestMutexArray.cpp
  TestPageTableManager.cpp
  TestPagingSystem.cpp
  TestPagingSystemKernels.cpp
  TestSparseTexture.cpp
  TestSparseTexture.cu
  TestSparseTexture.h
  TestSparseTextureWrap.cpp
  TestSparseVsDenseTextures.cpp
  TestSparseVsDenseTextures.cu
  TestSparseVsDenseTextures.h
  TestTextureFill.cpp
  TestTextureInstantiation.cpp
  TestTicket.cpp
  TestTileIndexing.cpp
  SourceDir.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/SourceDir.h
  )

source_group( "CUDA Files" REGULAR_EXPRESSION ".*\.cu" )

target_include_directories( testDemandLoading PUBLIC
  ../src
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>  # for SourceDir.h
  )

# Embed CMAKE_CURRENT_SOURCE_DIR in getSourceDir()
configure_file( SourceDir.h.in include/SourceDir.h @ONLY )

target_link_libraries( testDemandLoading
  testDemandLoadingKernels
  DemandLoadingKernels
  OptiX::OptiX
  DemandLoading
  ShaderUtil
  CUDA::cudart
  OpenEXR::OpenEXR # for half
  GTest::gmock
  ${CMAKE_DL_LIBS}
  )

if( OTK_USE_CUDA_MEMORY_POOLS )
  target_compile_definitions( testDemandLoading PRIVATE OTK_USE_CUDA_MEMORY_POOLS )
endif()

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:testDemandLoading> )
  add_custom_command( TARGET testDemandLoading POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:testDemandLoading> $<TARGET_FILE_DIR:testDemandLoading>
    COMMAND_EXPAND_LISTS )
endif()

set_target_properties( testDemandLoading 
  PROPERTIES CXX_STANDARD 14 # Required by latest gtest
  FOLDER DemandLoading/Tests
)


# Register test cases with CTest.
gtest_discover_tests(testDemandLoading PROPERTIES LABELS DemandLoading)

# The texture footprint test employs an OptiX kernel, which is compiled from CUDA to OPTIXIR.
include( embed_cuda )
embed_cuda(
  CONST HEADER TextureFootprintCuda.h
  OUTPUT_TARGET
    testTextureFootprintKernel
  LIBRARIES
    DemandLoading
  FOLDER DemandLoading/Tests
  SOURCES
    TestTextureFootprint.cu
)

otk_add_executable( testTextureFootprint
  TestTextureFootprint.cpp
  TestTextureFootprint.h
  )

target_include_directories( testTextureFootprint PUBLIC
  ../src
  )

target_link_libraries( testTextureFootprint
  testTextureFootprintKernel
  DemandLoading
  CUDA::cudart
  GTest::gtest_main
  ${CMAKE_DL_LIBS}
  )
set_target_properties( testTextureFootprint PROPERTIES FOLDER DemandLoading/Tests )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:testTextureFootprint> )
  add_custom_command( TARGET testTextureFootprint POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:testTextureFootprint> $<TARGET_FILE_DIR:testTextureFootprint>
    COMMAND_EXPAND_LISTS )
endif()

set_target_properties( testDemandLoading PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER DemandLoading/Tests
)

# Register test cases with CTest.
gtest_discover_tests(testTextureFootprint PROPERTIES LABELS DemandLoading)
