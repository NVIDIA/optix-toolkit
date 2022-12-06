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
// (INCLUDING NEGLIGENCE O00000R OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once 

#include <cuda_runtime.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/color.h>

struct RayPayload;

namespace ommBakingApp
{

//------------------------------------------------------------------------------
// OptiX utility functions
//------------------------------------------------------------------------------

// Get the pixel index for the current sample (horizontal striping over GPUs)
static __forceinline__ __device__ uint2 getPixelIndex( unsigned int num_devices, unsigned int device_index )
{
    const uint3 launchIndex = optixGetLaunchIndex();
    return make_uint2( launchIndex.x, ( launchIndex.y * num_devices + device_index ) );
}

// Return whether a pixel index is in bounds
static __forceinline__ __device__ bool pixelInBounds( uint2 p, unsigned int width, unsigned int height )
{
    return ( p.x < width && p.y < height );
}

// Convert 2 unsigned ints to a void*
static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void*                    ptr  = reinterpret_cast<void*>( uptr );
    return ptr;
}

// Convert a void* to 2 unsigned ints
static __forceinline__ __device__ void packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0                            = uptr >> 32;
    i1                            = uptr & 0x00000000ffffffff;
}

// Get the per-ray data for the current ray
static __forceinline__ __device__ RayPayload* getRayPayload()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RayPayload*>( unpackPointer( u0, u1 ) );
}

// Pack a float3 as ints for reporting an intersection in OptiX
#define float3_as_uints( u ) __float_as_uint( u.x ), __float_as_uint( u.y ), __float_as_uint( u.z )

// Trace a ray
static __forceinline__ __device__ void traceRay( OptixTraversableHandle handle,
                                                 RayType                ray_type,
                                                 float3                 ray_origin,
                                                 float3                 ray_direction,
                                                 float                  tmin,
                                                 float                  tmax,
                                                 float                  ray_time,
                                                 RayPayload*            ray_payload )
{
    unsigned int u0, u1;
    packPointer( ray_payload, u0, u1 );
    optixTrace( handle,                                           // traversable handle
                ray_origin, ray_direction, tmin, tmax, ray_time,  // ray definition
                OptixVisibilityMask( 1 ),                         // visibility mask
                OPTIX_RAY_FLAG_NONE,                              // flags
                ray_type,                                         // SBT offset
                RAY_TYPE_COUNT,                                   // SBT stride
                RAY_TYPE_RADIANCE,                                // missSBTIndex
                u0, u1                                            // Ray payload pointer
                );
}

} // namespace ommBakingApp