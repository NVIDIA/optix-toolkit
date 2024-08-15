# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

add_library(tinyddsloader INTERFACE)
target_include_directories(tinyddsloader INTERFACE .)
if(MSVC)
  # Work around warnings in tinyddsloader
  # warning C4702: unreachable code
  target_compile_options(tinyddsloader INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/wd4702>)
endif()

add_library(tinyexr INTERFACE)
target_include_directories(tinyexr INTERFACE .)

add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE .)
