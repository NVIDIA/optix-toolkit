// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <optix.h>

#include <cuda.h>

// for ::size_t
#include <stddef.h>

namespace demandPbrtScene {

struct Traversable
{
    CUdeviceptr            accelBuffer;
    size_t                 accelBufferSize;
    OptixTraversableHandle handle;
};

}  // namespace demandPbrtScene
