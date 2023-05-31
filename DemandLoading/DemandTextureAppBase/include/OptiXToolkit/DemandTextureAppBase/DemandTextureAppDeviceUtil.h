//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>

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
static __forceinline__ __device__ bool pixelInBounds( uint2 p, uint2 dim )
{
    return ( p.x < dim.x && p.y < dim.y );
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

// Get triangle barycentrics in a slightly more convenient form than OptiX gives them
static __forceinline__ __device__ float3 getTriangleBarycentrics()
{
    const float2 bary = optixGetTriangleBarycentrics();
    return float3{1.0f - bary.x - bary.y, bary.x, bary.y};
}

// Pack a float3 as ints for reporting an intersection in OptiX
#define float3_as_uints( u ) __float_as_uint( u.x ), __float_as_uint( u.y ), __float_as_uint( u.z )

// Trace a ray
static __forceinline__ __device__ void traceRay( OptixTraversableHandle handle,
                                                 float3                 ray_origin,
                                                 float3                 ray_direction,
                                                 float                  tmin,
                                                 float                  tmax,
                                                 uint32_t               ray_flags,
                                                 void*                  ray_payload )
{

    unsigned int u0, u1;
    packPointer( ray_payload, u0, u1 );
    const float ray_time = 0.0f;
    optixTrace( handle,                                           // traversable handle
                ray_origin, ray_direction, tmin, tmax, ray_time,  // ray definition
                OptixVisibilityMask( 1 ),                         // visibility mask
                ray_flags,                                        // flags
                RAY_TYPE_RADIANCE,                                // SBT offset
                RAY_TYPE_COUNT,                                   // SBT stride
                RAY_TYPE_RADIANCE,                                // missSBTIndex
                u0, u1                                            // Ray payload pointer
                );
}

// Determine if the current ray is an occlusion ray
static __forceinline__ __device__ unsigned int isOcclusionRay() 
{
    return ( optixGetRayFlags() & OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT );
}

// Concentric mapping from Ray Tracing Gems : Sampling Transformations Zoo
static __forceinline__ __device__ float2 concentricMapping( float2 u )
{
    float a = 2.0f * u.x - 1.0f; 
    float b = 2.0f * u.y - 1.0f;
    if ( b == 0.0f ) b = 1.0f;
    float r, phi;

    if ( a * a > b * b ) 
    {
        r = a;
        phi = ( M_PIf / 4.0f ) * ( b / a );
    } 
    else 
    {
        r = b;
        phi = ( M_PIf / 2.0f ) - ( M_PIf / 4.0f ) * ( a / b );
    }
    return float2{r * cos( phi ), r * sin( phi )};
}

// Make an orthonormal basis from unit length vector N.
static __forceinline__ __device__ void makeOrthoBasis( float3 N, float3& S, float3& T )
{
    using namespace otk;
    S = ( fabsf( N.x ) < fabsf( N.y ) ) ? float3{-N.y, N.x, 0.0f} : float3{0.0f, -N.z, N.y};
    S = normalize( S );
    T = cross( S, N );
}

// Seed a random number generator for a given pixel and launch index
static __forceinline__ __device__ unsigned int srand( unsigned int x, unsigned int y, unsigned int z )
{
    const unsigned int xyz = 84721u * x + 28411u * y + 339341u * z + 23478;
    return xyz + xyz * xyz * xyz;
}

// get a random number in [0,1)
static __forceinline__ __device__ float rnd( unsigned int& prev )
{
    prev = 1664525u * prev + 1013904223u;
    return static_cast<float>( prev & 0x00FFFFFF ) / 0x01000000;
}

// transform a point in [0,1) to a tent filter distribution in [-0.5, 1.5)
static __forceinline__ __device__ float tentFilter( float x )
{
    return ( x < 0.5f ) ? -0.5f + sqrtf( x * 2.0f ) : 1.5f - sqrtf( 2.0f - x * 2.0f );
}

//------------------------------------------------------------------------------
// Camera rays
//------------------------------------------------------------------------------

// Get eye ray for orthographic camera, where par.U, par.V are semi-axes of view
static __forceinline__ __device__ 
void makeEyeRayOrthographic( CameraFrame camera, uint2 image_dim, float2 px, float3& origin, float3& direction )
{
    using namespace otk;
    origin = camera.eye; 
    origin += ( 2.0f * ( px.x - 0.5f * image_dim.x ) / image_dim.x ) * camera.U;
    origin += ( 2.0f * ( px.y - 0.5f * image_dim.y ) / image_dim.y ) * camera.V;
    direction = normalize( camera.W );
}

// Get eye ray for pinhole camera, where par.U, par.V are semi-axes of view
static __forceinline__ __device__ 
void makeEyeRayPinhole( CameraFrame camera, uint2 image_dim, float2 px, float3& origin, float3& direction )
{
    using namespace otk;
    origin = camera.eye;
    direction = camera.W;
    direction += ( 2.0f * ( px.x - 0.5f * image_dim.x ) / image_dim.x ) * camera.U;
    direction += ( 2.0f * ( px.y - 0.5f * image_dim.y) / image_dim.y ) * camera.V;
    direction = normalize( direction );
}

// Get eye ray for a thin lens camera, where  where par.U, par.V are semi-axes of view
static __forceinline__ __device__ 
void makeCameraRayThinLens( CameraFrame camera, float lens_width, uint2 image_dim, float2 lx, float2 px, float3& origin, float3& direction )
{
    using namespace otk;
    // Calculate ray origin offset from eye point, (assuming lens at origin)
    float2 luv = concentricMapping( lx );
    origin = float3{0.0f, 0.0f, 0.0f};
    origin += ( 0.5f * lens_width * luv.x / length( camera.U ) ) * camera.U;
    origin += ( 0.5f * lens_width * luv.y / length( camera.V ) ) * camera.V;

    // Ray direction is vector from lens point to a point on the image plane
    direction = camera.W;
    direction += ( 2.0f * ( px.x - 0.5f * image_dim.x ) / image_dim.x ) * camera.U;
    direction += ( 2.0f * ( px.y - 0.5f * image_dim.y ) / image_dim.y ) * camera.V;
    direction -= origin;
    direction = normalize( direction );

    // Add the eye point to the origin
    origin += camera.eye;
}

//------------------------------------------------------------------------------
// Texture sampling and display
//------------------------------------------------------------------------------

// Sample demand-load texture, walking up the mip pyramid to find a resident sample
template <class TYPE> 
static __forceinline__ __device__ 
TYPE tex2DGradWalkup( const DeviceContext context, unsigned int texture_id, float x, float y, float2 ddx, float2 ddy, bool* is_resident )
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

// Sample a ColorTex
static __forceinline__ __device__ 
float3 sampleTexture( const DeviceContext context, SurfaceGeometry g, ColorTex tex, bool* is_resident )
{
    using namespace otk;
    if( tex.texid < 0 )
        return tex.color;
    float4 t = tex2DGradWalkup<float4>( context, tex.texid, g.uv.x, g.uv.y, g.ddx, g.ddy, is_resident );
    return tex.color * float3{t.x, t.y, t.z};
}

// For pixel px, compute the color of an overlay showing resident texture tiles graphically,
// where the overlay display is located at (x0, y0).
static __forceinline__ __device__ float4 tileDisplayColor( const DeviceContext& context, int texture_id, int x0, int y0, uint2 px )
{
    using namespace otk;
    if( texture_id < 0 )
        return make_float4( 0.0f );

    // Get the texture sampler
    bool resident;
    unsigned long long samplerPage = pagingMapOrRequest( context, texture_id, &resident );
    if( samplerPage == 0 )
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
