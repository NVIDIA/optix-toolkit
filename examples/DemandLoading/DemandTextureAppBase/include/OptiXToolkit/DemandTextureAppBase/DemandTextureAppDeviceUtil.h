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

#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/Paging.h>
#include <OptiXToolkit/DemandLoading/Texture2DExtended.h>
#include <OptiXToolkit/DemandLoading/TextureSampler.h>

using namespace demandLoading;
struct RayPayload;

namespace demandTextureApp
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


//------------------------------------------------------------------------------
// Ray cones
//------------------------------------------------------------------------------

// Propagate a ray cone through distance
static __forceinline__ __device__ float propagateRayCone( float cone_width, float cone_angle, float distance )
{
    return fabsf( cone_width + cone_angle * distance );
}

// Compute a conservative bound (at least as small as) the minimum width in texture space that a ray cone projects to.
// The resulting mip level does not over-blur the texture, but anisotropy is not handled.
static __forceinline__ __device__ float computeTextureFootprintMinWidth( float dPds_len, float dPdt_len, float cone_width )
{
    return cone_width / maxf( dPds_len, dPdt_len );
}

//------------------------------------------------------------------------------
// Texture sampling and display
//------------------------------------------------------------------------------

// Sample demand-load texture, walking up the mip pyramid to find a resident sample
template <class TYPE>
static __forceinline__ __device__ TYPE
tex2DGradWalkup( const DeviceContext& context, unsigned int texture_id, float x, float y, float2 ddx, float2 ddy, bool* is_resident )
{
    const float mipTailSampleWidth = 1.0f / 32.0f;
    const int   maxWalkup          = 4;
    bool        resident           = false;
    TYPE        color;

    for( int i = 0; i < maxWalkup; ++i )
    {
        color        = tex2DGradUdimBlend<TYPE>( context, texture_id, x, y, ddx, ddy, &resident );
        *is_resident = *is_resident && resident;
        if( resident )
            return color;

        // Double the texture gradients at each iteration, but request the mip tail the last time.
        ddx = (i < maxWalkup - 2) ? ddx * 2.0f : float2{ mipTailSampleWidth, 0.0f };
        ddy = (i < maxWalkup - 2) ? ddy * 2.0f : float2{ 0.0f, mipTailSampleWidth };
    }

    // If all texture samples fail, return the base color
    bool baseColorValid = getBaseColor<TYPE>( context, texture_id, color, &resident );
    return color;
}

// For pixel px, compute the color of an overlay showing resident texture tiles graphically,
// where the overlay display is located at (x0, y0).
static __forceinline__ __device__ float4 tileDisplayColor( const DeviceContext& context, int texture_id, int x0, int y0, uint2 px )
{
    if( texture_id < 0 )
        return make_float4( 0.0f );

    // Get the texture sampler
    bool resident;
    unsigned long long samplerPage = pagingMapOrRequest( context, texture_id, &resident );
    if( !resident )
        return make_float4( 0.0f );
    demandLoading::TextureSampler* info = reinterpret_cast<demandLoading::TextureSampler*>( samplerPage );

    // For each mip level, check to see if px is inside its display rectangle
    int lastLevel = min( info->desc.numMipLevels - 1, info->mipTailFirstLevel );
    for( int level = 0; level <= lastLevel; ++level )
    {
        unsigned int widthInTiles = demandLoading::getLevelDimInTiles( info->width, level, 1 << info->desc.logTileWidth );
        unsigned int heightInTiles = demandLoading::getLevelDimInTiles( info->height, level, 1 << info->desc.logTileHeight );
        int          x1            = x0 + widthInTiles - 1;
        int          y1            = y0 + heightInTiles - 1;

        // If px is inside the rectangle for this level, return a color based on value in the LRU table
        if( px.x >= x0 && px.x <= x1 && px.y >= y0 && px.y <= y1 )
        {
            int tileOffset = getPageOffsetFromTileCoords( px.x - x0, px.y - y0, widthInTiles );
            int pageId     = info->startPage + info->mipLevelSizes[level].mipLevelStart + tileOffset;

            if( !demandLoading::checkBitSet( pageId, context.residenceBits ) )
                return make_float4( 0.2f, 0.2f, 0.2f, 0.9f );

            float lruVal = ( context.lruTable ) ? ( demandLoading::getHalfByte( pageId, context.lruTable ) / 16.0f ) : 0.0f;
            return make_float4( lruVal, 1.0f - lruVal, 0.0f, 0.9f );
        }
        x0 = x1 + 5;
    }
    return make_float4( 0.0f );
}

} // namespace demandTextureApp