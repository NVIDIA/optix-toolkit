# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(FetchGtest)
include(GoogleTest)
include(embed_cuda)

# The omm baking test employs an OptiX kernel, which is compiled from CUDA to OptiXIR.
embed_cuda(
  CONST HEADER testCuOmmBakingKernelsCuda.h
  OUTPUT_TARGET
    testCuOmmBakingKernels
  LIBRARIES
    CuOmmBaking
  SOURCES
    Util/OptiXKernels.cu
    Util/OptiXKernels.h
  FOLDER
    OmmBaking/Tests
)

add_definitions( -DTEST_OMM_BAKING_TEXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Textures/" )
add_definitions( -DTEST_OMM_BAKING_GOLD_DIR="${CMAKE_CURRENT_SOURCE_DIR}/gold/" )

otk_add_executable( testCuOmmBaking
  cuOmmBakingErrorCheck.h
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
  SourceDir.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/SourceDir.h
  )  

set_target_properties(testCuOmmBaking PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER OmmBaking/Tests 
  )

target_include_directories( testCuOmmBaking PUBLIC
  ../src
  ${CMAKE_CURRENT_BINARY_DIR}/include  # for SourceDir.h
  )

# Embed CMAKE_CURRENT_SOURCE_DIR in getSourceDir()
configure_file( SourceDir.h.in include/SourceDir.h @ONLY )

target_link_libraries( testCuOmmBaking
  GTest::gtest_main
  testCuOmmBakingKernels
  CuOmmBaking
  OptiX::OptiX
  OptiXToolkit::Error
  OptiXToolkit::OptiXMemory
  tinyddsloader
  tinyexr
  tinygltf
  )

if(NOT MSVC)
  # Work around warnings in stb_image and tinyddsloader
  target_compile_options(testCuOmmBaking PRIVATE -Wno-type-limits -Wno-switch)
endif()

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:testCuOmmBaking> )
  add_custom_command( TARGET testCuOmmBaking POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:testCuOmmBaking> $<TARGET_FILE_DIR:testCuOmmBaking>
    COMMAND_EXPAND_LISTS )
endif()

# Register test cases with CTest.
gtest_discover_tests(testCuOmmBaking PROPERTIES LABELS OmmBaking)
