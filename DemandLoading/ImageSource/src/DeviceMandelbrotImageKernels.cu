// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
