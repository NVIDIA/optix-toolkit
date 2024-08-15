// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

struct DeviceConstantImageParams
{
    unsigned int num_pixels;
    float4 color;
    float4* output_buffer;
};
