# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include( FetchGtest )
include( GoogleTest )

otk_add_executable( testNeuralTextures
  TestNeuralTextureSource.cpp
  SourceDir.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/SourceDir.h
)

target_link_libraries( testNeuralTextures PUBLIC
    NeuralTextures
    OptiXToolkit::ImageSource
    OptiXToolkit::NeuralTextures
    GTest::gmock_main
    CUDA::cudart
)

# Embed CMAKE_CURRENT_SOURCE_DIR in getSourceDir()
configure_file( SourceDir.h.in include/SourceDir.h @ONLY )

target_include_directories( testNeuralTextures PUBLIC 
  ${CMAKE_CURRENT_BINARY_DIR}/include )

set_target_properties( testNeuralTextures PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER NeuralTextures/Tests
)

# Single CTest entry per binary: this binary creates a CUDA/OptiX context that
# is reused across cases, so one process pays context init once, not per case.
add_test( NAME testNeuralTextures COMMAND testNeuralTextures )
set_tests_properties( testNeuralTextures PROPERTIES LABELS NeuralTextures )

