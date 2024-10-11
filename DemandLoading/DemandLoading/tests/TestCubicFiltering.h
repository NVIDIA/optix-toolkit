// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <OptiXToolkit/DemandLoading/DeviceContext.h>

__host__ void launchCubicTextureDrawKernel( CUstream stream, demandLoading::DeviceContext& context, unsigned int textureId,
                                            float4* output, int width, int height, float2 ddx, float2 ddy );

__host__ void launchCubicTextureSubimageDrawKernel( CUstream stream, demandLoading::DeviceContext& context,
                                                    unsigned int textureId, float4* image, float4* drdsImage, int width, int height,
                                                    float2 uv00, float2 uv11, float2 ddx, float2 ddy );
