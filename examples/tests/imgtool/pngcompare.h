// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <string>

struct OptPngCompare
{
  std::string file1;
  std::string file2;
  std::string diffFile;
  float       diffThreshold     = 0.0f;
  float       allowedPercentage = 0.0f;
};

int pngcompare( const OptPngCompare& opts );
