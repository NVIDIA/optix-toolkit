// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "TestSparseVsDenseTextures.h"

#include <cstdio>

#if __CUDA_ARCH__ >= 600
#define SPARSE_TEX_SUPPORT true
#endif

__global__ static void sparseVsDenseTextureKernel( CUtexObject texture )
{
    bool   isResident = true;
    float  x          = 0.5f;
    float  y          = 0.5f;
    float2 ddx        = make_float2(0.0f, 0.0f);
    float2 ddy        = make_float2(0.0f, 0.0f);
    
    // This should print 0 (0 0 0) since no tiles were loaded, but prints 1 (0 0 0)
#ifdef SPARSE_TEX_SUPPORT
    float4 val = tex2DGrad<float4>( texture, x, y, ddx, ddy, &isResident );
#else 
    float4 val = tex2DGrad<float4>( texture, x, y, ddx, ddy );
#endif

    printf("%d (%1.1f %1.1f %1.1f)\n", isResident, val.x, val.y, val.z);
}

__host__ void launchSparseVsDenseTextureKernel( CUtexObject texture )
{
    dim3 dimBlock( 1 );
    dim3 dimGrid( 1 );
    sparseVsDenseTextureKernel<<<dimGrid, dimBlock>>>( texture );
}
