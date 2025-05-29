// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

__host__ void launchDrawTextureKernel( CUstream stream, float4* image, int width, int height, unsigned long long texObj,
                                       float2 uv00, float2 uv11, float2 ddx, float2 ddy );
