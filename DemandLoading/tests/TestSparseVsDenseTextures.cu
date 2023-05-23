//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "TestSparseVsDenseTextures.h"

#include "Util/Exception.h"

__global__ static void sparseVsDenseTextureKernel( CUtexObject texture )
{
    bool   isResident = false;
    float  x          = 0.5f;
    float  y          = 0.5f;
//    float2 ddx        = make_float2(0.1f, 0.0f);
//    float2 ddy        = make_float2(0.0f, 0.1f);
    float2 ddx        = make_float2(0.0f, 0.0f);
    float2 ddy        = make_float2(0.0f, 0.0f);
    
    // This should print 0 (0 0 0) since no tiles were loaded, but prints 1 (0 0 0)
    float4 val = tex2DGrad<float4>( texture, x, y, ddx, ddy, &isResident );
    printf("%d (%1.1f %1.1f %1.1f)\n", isResident, val.x, val.y, val.z);
}

__host__ void launchSparseVsDenseTextureKernel( CUtexObject texture )
{
    dim3 dimBlock( 1 );
    dim3 dimGrid( 1 );
    sparseVsDenseTextureKernel<<<dimGrid, dimBlock>>>( texture );
}
