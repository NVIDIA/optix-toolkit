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

#include <OptiXToolkit/ImageSource/DeviceMandelbrotParams.h>

namespace imageSource {

extern "C" __global__ void deviceReadMandelbrotImage(const MandelbrotParams params)
{
    int     idx          = blockIdx.x * blockDim.x + threadIdx.x;
    int     width        = params.clip_width;
    int     height       = params.clip_height;
    float4* outputBuffer = params.output_buffer;

    // Figure out which mip level idx is in
    while( ( params.all_mip_levels == true ) && ( idx >= width * height ) && ( width > 1 || height > 1 ) )
    {
        idx -= width * height;
        outputBuffer += width * height;
        width  = ( width == 1 ) ? 1 : width >> 1;
        height = ( height == 1 ) ? 1 : height >> 1;
    }

    // Return if the index is out of image
    if( idx >= width * height )
        return;

    int j = idx / width;
    int i = idx - ( j * width );
    
    float4 c{0.0f, 0.0f, 0.0f, 0.0f};

    // If inside the mandelbrot image (not on a tile border), get the mandelbrot color 
    if( params.all_mip_levels || (i < params.clip_width && j < params.clip_height) )
    {
        double dx = static_cast<double>(i + 0.5) / width;
        double x  = params.xmin * (1.0-dx) + params.xmax * dx;

        double dy = static_cast<double>(j + 0.5) / height;
        double y  = params.ymin * (1.0-dy) + params.ymax * dy;
        
        c = mandelbrotColor( x, y, params );
    }
    
    outputBuffer[idx] = c;
}

__host__ void launchReadMandelbrotImage( const MandelbrotParams& params, CUstream stream )
{
    unsigned int totalThreads = params.width * params.height;
    if( params.all_mip_levels == true )
        totalThreads = totalThreads * 4 / 3;
    
    const unsigned int threadsPerBlock = 64;
    const unsigned int numBLocks       = ( totalThreads + threadsPerBlock - 1 ) / threadsPerBlock;

    dim3 grid( numBLocks, 1, 1 );
    dim3 block( threadsPerBlock, 1, 1 );

    deviceReadMandelbrotImage<<<grid, block, 0U, stream>>>( params );
}

} // namespace imageSource
