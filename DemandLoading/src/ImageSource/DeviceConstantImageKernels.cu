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

#include <cuda.h>
#include <cuda_runtime.h>

#include <ImageSource/DeviceConstantImageParams.h>

namespace imageSource {

extern "C" __global__ void deviceReadConstantImage(const DeviceConstantImageParams params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx >= params.num_pixels )
        return;
    
    if( idx < 64 || (idx & 63) == 0 )
        params.output_buffer[idx] = float4{0.0f, 0.0f, 0.0f, 0.0f};
    else
        params.output_buffer[idx] = params.color;
}

__host__ void launchReadConstantImage( const DeviceConstantImageParams& params, CUstream stream )
{
    unsigned int       totalThreads    = params.num_pixels;
    const unsigned int threadsPerBlock = 64;
    const unsigned int numBLocks       = ( totalThreads + threadsPerBlock - 1 ) / threadsPerBlock;

    dim3 grid( numBLocks, 1, 1 );
    dim3 block( threadsPerBlock, 1, 1 );

    deviceReadConstantImage<<<grid, block, 0U, stream>>>( params );
}

} // namespace imageSource