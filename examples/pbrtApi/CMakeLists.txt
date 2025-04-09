# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

add_library( pbrtApi STATIC
  # These source files provide the missing pbrt functions that bridge
  # the API to an interface.
  PbrtApi.cpp
  include/OptiXToolkit/PbrtApi/PbrtApi.h
  ReadMe.md
  )
source_group("Source Files" REGULAR_EXPRESSION "[^/]*\\.cpp")
source_group("Header Files" REGULAR_EXPRESSION "[^/]*\\.h")

set_property(TARGET pbrtApi PROPERTY FOLDER "Examples/Libraries")

target_include_directories( pbrtApi PUBLIC include )
target_link_libraries(pbrtApi PUBLIC pbrtParser )
