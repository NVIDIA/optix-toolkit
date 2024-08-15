// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda_runtime.h>

struct Params
{
    uchar4* image;
    unsigned int image_width;
};

struct RayGenData
{
    float r,g,b;
};
