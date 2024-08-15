# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

if(POLICY CMP0048)
  # The project() command manages VERSION variables.
  cmake_policy(SET CMP0048 NEW)
endif()

if(POLICY CMP0074)
  # find_package uses <PackageName>_ROOT variables.
  cmake_policy(SET CMP0074 NEW)
endif()
