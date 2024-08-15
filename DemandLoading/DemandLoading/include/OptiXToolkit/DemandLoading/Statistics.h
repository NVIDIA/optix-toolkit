// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cstddef>

namespace demandLoading {

struct Statistics
{
    // Stats that are shared between devices
    size_t numTilesRead;
    size_t numBytesRead;
    double readTime;
    double requestProcessingTime;
    size_t numTextures;
    size_t virtualTextureBytes;

    // Per-device stats
    size_t deviceMemoryUsed;
    size_t bytesTransferredToDevice;
    unsigned int numEvictions;
};

}  // namespace demandLoading
