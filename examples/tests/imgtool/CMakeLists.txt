# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(BuildConfig)
include(FetchStbImage)

otk_add_executable(imgtool 
  imgtool.cpp
  pngcompare.cpp
  pngcompare.h
  )
target_link_libraries(imgtool PUBLIC Stb::Image)
set_target_properties(imgtool PROPERTIES FOLDER Examples/Tests)
