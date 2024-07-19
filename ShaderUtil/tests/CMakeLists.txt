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

if(NOT TARGET GTest::gtest)
    find_package( GTest QUIET )
    if(NOT GTEST_FOUND)
      message(WARNING "GTest not found.  Skipping ShaderUtil/tests.")
      return()
    endif()
endif()

include(GoogleTest)

if(NOT TARGET OptiX::OptiX)
  # Set OptiX_INSTALL_DIR to the root of the OptiX SDK when configuring CMake.
  set(OptiX_INSTALL_DIR "OptiX_INSTALL_DIR-NOTFOUND" CACHE PATH "Path to OptiX installed location.")
  find_package(OptiX 7.3 QUIET)
  if(NOT OPTIX_FOUND)
    message(WARNING "OptiX not found.  Skipping ShaderUtil/tests.")
    return()
  endif()  
endif()

# The SIA test employs an OptiX kernel, which is compiled from CUDA to OptiXIR.
include(embed_cuda OPTIONAL RESULT_VARIABLE EMBED_CUDA_FOUND)
if(NOT EMBED_CUDA_FOUND)
  message(WARNING "embed_cuda.cmake not found.  Skipping ShaderUtil/tests.")
  return()
endif()  

embed_cuda(
  PTX
  CONST HEADER testSiaKernelsCuda.h
  OUTPUT_TARGET
    testSelfIntersectionAvoidanceKernels
  LIBRARIES
    OptiXToolkit::ShaderUtil
    OptiX::OptiX
  SOURCES
    testSiaOptix.cu
  FOLDER ShaderUtil/Tests
)

# Embed CMAKE_CURRENT_SOURCE_DIR in getSourceDir()
configure_file( SourceDir.h.in include/SourceDir.h @ONLY )
otk_add_executable( testSelfIntersectionAvoidance
  testSia.h
  testSia.cpp
  testSiaCuda.cu
  SourceDir.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/SourceDir.h
  )
set_target_properties(testSelfIntersectionAvoidance PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER ShaderUtil/Tests 
  )
target_include_directories( testSelfIntersectionAvoidance PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/include  # for SourceDir.h
  )
target_link_libraries( testSelfIntersectionAvoidance
  GTest::gtest_main
  testSelfIntersectionAvoidanceKernels
  OptiXToolkit::ShaderUtil
  OptiX::OptiX
  )
# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:testSelfIntersectionAvoidance> )
  add_custom_command( TARGET testSelfIntersectionAvoidance POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:testSelfIntersectionAvoidance> $<TARGET_FILE_DIR:testSelfIntersectionAvoidance>
    COMMAND_EXPAND_LISTS )
endif()
# Register test cases with CTest.
gtest_discover_tests(testSelfIntersectionAvoidance PROPERTIES LABELS ShaderUtil)


embed_cuda(
  CONST HEADER TestShaderUtilsCuda.h
  OUTPUT_TARGET
    TestShaderUtilsKernels
  LIBRARIES
    OptiXToolkit::ShaderUtil
    OptiX::OptiX
  SOURCES
    TestDebugLocation.cu
  FOLDER ShaderUtil/Tests
)

add_executable(TestShaderUtil
    TestAliasTable.cpp
    TestCdfInversionTable.cpp
    TestDebugLocation.h
    TestDebugLocation.cpp
    TestDebugLocationParams.h
    TestOperators.cpp
    TestPdfTable.cpp
    TestPrinters.cpp
    TestISummedAreaTable.cpp
    TestTransform4.cpp
)
target_link_libraries(TestShaderUtil PUBLIC
    TestShaderUtilsKernels
    GTest::gmock_main
    OptiXToolkit::Error
    OptiXToolkit::Memory
    OptiXToolkit::OptiXMemory
    OptiXToolkit::ShaderUtil
    OptiX::OptiX
)
set_target_properties(TestShaderUtil PROPERTIES FOLDER ShaderUtil/Tests)
if( $<TARGET_RUNTIME_DLLS:TestShaderUtil> )
  add_custom_command( TARGET TestShaderUtil POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:TestShaderUtil> $<TARGET_FILE_DIR:TestShaderUtil>
    COMMAND_EXPAND_LISTS )
endif()
gtest_discover_tests(TestShaderUtil PROPERTIES LABELS ShaderUtil)
