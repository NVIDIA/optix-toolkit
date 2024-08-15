// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

namespace imageSource {

struct CacheStatistics
{
    unsigned int       numImageSources;
    unsigned long long totalTilesRead;
    unsigned long long totalBytesRead;
    double             totalReadTime;
};

}  // namespace imageSource
