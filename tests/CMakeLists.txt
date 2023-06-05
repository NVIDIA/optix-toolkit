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

include(FetchGtest)
include(GoogleTest)

# The omm baking test employs an OptiX kernel, which is compiled from CUDA to PTX.
include(embed_ptx)
embed_ptx(
  CONST HEADER testCuOmmBakingKernelsPTX.h
  OUTPUT_TARGET
    testCuOmmBakingKernels
  EMBED_PTX_PTX_TARGET
    testCuOmmBakingKernels_ptx
  PTX_LINK_LIBRARIES
    CuOmmBaking
  SOURCES
    Util/OptiXKernels.cu
    Util/OptiXKernels.h
)

set_target_properties(testCuOmmBakingKernels testCuOmmBakingKernels_ptx PROPERTIES FOLDER CmOmmBaking/tests )

add_definitions( -DTEST_OMM_BAKING_TEXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Textures/" )
add_definitions( -DTEST_OMM_BAKING_GOLD_DIR="${CMAKE_CURRENT_SOURCE_DIR}/gold/" )

otk_add_executable( testCuOmmBaking
  testCommon.h
  testCommon.cpp
  testCuOmmBaking.cpp
  testInvalidInput.cpp
  Util/BakeTexture.cu
  Util/BakeTexture.h
  Util/Image.cpp
  Util/Image.h
  Util/Mesh.cpp
  Util/Mesh.h
  Util/OptiXKernels.h
  Util/OptiXOmmArray.cpp
  Util/OptiXOmmArray.h
  Util/OptiXScene.cpp
  Util/OptiXScene.h  
  )  

set_target_properties(testCuOmmBaking PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER CmOmmBaking/tests 
  )

target_include_directories( testCuOmmBaking PUBLIC
  ../support
  ../src
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>  # for SourceDir.h
  )

# Embed CMAKE_CURRENT_SOURCE_DIR in getSourceDir()
configure_file( SourceDir.h.in include/SourceDir.h @ONLY )

target_link_libraries( testCuOmmBaking
  GTest::gtest_main
  testCuOmmBakingKernels
  CuOmmBaking
  OptiX::OptiX
  )

if(NOT MSVC)
  # Work around warnings in stb_image and tinyddsloader
  target_compile_options(testCuOmmBaking PRIVATE -Wno-type-limits -Wno-switch) 
endif()

add_dependencies( testCuOmmBaking testCuOmmBakingKernels )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:testCuOmmBaking> )
  add_custom_command( TARGET testCuOmmBaking POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:testCuOmmBaking> $<TARGET_FILE_DIR:testCuOmmBaking>
    COMMAND_EXPAND_LISTS )
endif()

# Register test cases with CTest.
gtest_discover_tests(testCuOmmBaking)
