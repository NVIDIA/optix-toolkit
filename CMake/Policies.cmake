# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

if(POLICY CMP0048) # introduced in CMake 3.0
  # The project() command manages VERSION variables.
  cmake_policy(SET CMP0048 NEW)
endif()

if(POLICY CMP0074) # introduced in CMake 3.12
  # find_package uses <PackageName>_ROOT variables.
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0167)  # introduced in CMake 3.30
  cmake_policy(SET CMP0167 NEW) # ignore FindBoost and use BoostConfig
endif()

