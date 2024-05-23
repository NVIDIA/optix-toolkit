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
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

namespace demandLoading {

/// Get the dimensions of a demand load texture at a given texture coordinate, accounting for udims.
static __forceinline__ __device__ float2 getTextureDims( const DeviceContext& context, unsigned int texId, float2 uv )
{
    // Assume samplers are null if not resident
    const TextureSampler* sampler = reinterpret_cast<const TextureSampler*>( context.pageTable.data[texId] );
    if( !sampler )
        return float2{1.0f, 1.0f};

    // If not udim texture, return texture dims
    if( sampler->udim == 0 )
        return float2{static_cast<float>(sampler->width), static_cast<float>(sampler->height)};

    // Get subtexture and multiply texture dims by udims
    float        sx, sy;
    unsigned int xidx, yidx;
    wrapAndSeparateUdimCoord( uv.x, CU_TR_ADDRESS_MODE_WRAP, sampler->udim, sx, xidx );
    wrapAndSeparateUdimCoord( uv.y, CU_TR_ADDRESS_MODE_WRAP, sampler->vdim, sy, yidx );
    unsigned int subTexId = sampler->udimStartPage + ( yidx * sampler->udim + xidx );
    
    const TextureSampler* subSampler = reinterpret_cast<const TextureSampler*>( context.pageTable.data[subTexId] );
    if( !subSampler )
        return float2{static_cast<float>( sampler->width ), static_cast<float>( sampler->height )};
    return float2{ static_cast<float>( subSampler->width * sampler->udim ), static_cast<float>( sampler->height * sampler->vdim ) };
}

static __forceinline__ __device__ float sinc( float x )
{
    if( fabs( x ) < 0.0001f ) 
        return 1.0f;
    return sin( M_PI * x ) / ( M_PI * x );
}

static __forceinline__ __device__ float lanczos( float x, float a )
{
    if( x * x >= a * a )
        return 0.0f;
    return sinc( x ) * sinc( x / a ); 
}

static __forceinline__ __device__ float mitchell( float x, float b, float c )
{
    x = fabs(x);
    float fx = 0.0f;
    if( x < 1.0f )
        fx = (12.0f - 9.0f*b - 6.0f*c)*x*x*x + (-18.0f + 12.0f*b + 6.0f*c)*x*x + (6.0f - 2.0f*b);
    else if( x < 2.0f )
        fx = (-b - 6.0f*c)*x*x*x + (6.0f*b + 30.0f*c)*x*x + (-12.0f*b - 48.0f*c)*x + (8.0f*b + 24.0f*c);
    fx *= (1.0f / 6.0f);
    return fx;
}

/// Lanczos filter
template <class TYPE> static __forceinline__ __device__ TYPE
tex2DLanczos( const DeviceContext& context, unsigned int textureId, float x, float y, bool* isResident )
{
    // Scale x,y to integer coordinates, and get fractional offsets (fx,fy)
    float2 texDim = getTextureDims( context, textureId, float2{x, y} );
    x = x * texDim.x - 0.5f;
    y = y * texDim.y - 0.5f;
    float fx = x - floorf( x );
    float fy = y - floorf( y );

    // Take 16 weighted taps
    TYPE rval{};
    float weight = 0.0f;
    for( int j=-1; j<=2; j++)
    {
        float ly = lanczos( j - fy, 2.0f );
        float yy = (floorf( y + j ) + 0.5f) / texDim.y;

        for( int i=-1; i <= 2; i++)
        {
            float lx = lanczos( i - fx, 2.0f );
            float xx = ( floorf( x + i ) + 0.5f ) / texDim.x;

            TYPE tap = tex2D<TYPE>( context, textureId, xx, yy, isResident );
            weight += lx * ly;
            rval += ( lx * ly ) * tap;
        }
    }
    return rval * ( 1.0f / weight );
}

/// Mitchell filter
// Standard Mitchell suggested by Mitchell and Netravali: b=0.33333f, c=0.33333f
// Bicubic filter: b=1, c=0 
template <class TYPE> static __forceinline__ __device__ TYPE
tex2DMitchell( float b, float c, const DeviceContext& context, unsigned int textureId, float x, float y, bool* isResident )
{
    // Scale x,y to integer coordinates, and get fractional offsets (fx,fy)
    float2 texDim = getTextureDims( context, textureId, float2{x, y} );
    x = x * texDim.x - 0.5f;
    y = y * texDim.y - 0.5f;
    float fx = x - floorf( x );
    float fy = y - floorf( y );

    // Take 16 weighted taps
    TYPE rval{};
    float weight = 0.0f;
    for( int j=-1; j <= 2; j++)
    {
        float ly = mitchell( j - fy, b, c );
        float yy = ( floorf( y + j ) + 0.5f ) / texDim.y;

        for( int i=-1; i<=2; i++)
        {
            float lx = mitchell( i - fx, b, c );
            float xx = ( floorf( x + i ) + 0.5f ) / texDim.x;

            TYPE tap = tex2D<TYPE>( context, textureId, xx, yy, isResident );
            weight += lx * ly;
            rval += ( lx * ly ) * tap;
        }
    }
    return rval * ( 1.0f / weight );
}

} // namespace demandLoading