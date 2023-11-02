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

static __forceinline__ __device__ float4 cubicWeights( float x )
{
    float x2 = x * x;
    float x3 = x2 * x;
    float4 weight;
    weight.x = -x3 + 3.0f*x2 - 3.0f*x + 1.0f,
    weight.y = 3.0f*x3 - 6.0f*x2 + 4.0f,
    weight.z = -3.0f*x3 + 3.0f*x2 + 3.0f*x + 1.0f,
    weight.w = x3;
    return weight * ( 1.0f / 6.0f );
}

// Cubic upsampling using 4 taps, based on the technique in GPU Gems 2, chapter 20: Fast Third-Order Texture Filtering.
template <class TYPE> static __forceinline__ __device__ TYPE
tex2DCubic( const DeviceContext& context, unsigned int textureId, float x, float y, bool* isResident )
{
    const TextureSampler* sampler = reinterpret_cast<const TextureSampler*>( pagingMapOrRequest( context, textureId, isResident ) );
    if( !sampler ) 
        return TYPE{};

    const float invTexWidth = 1.0f / sampler->width;
    const float invTexHeight = 1.0f / sampler->height;

    // Scale x,y to integer coordinates, and get fractional offsets (fx,fy)
    x = x * sampler->width - 0.5f;
    y = y * sampler->height - 0.5f;
    float fx = x - floorf( x );
    float fy = y - floorf( y );
    x -= fx;
    y -= fy;

    // Compute cubic weights, and precompute paired sums
    float4 wx = cubicWeights( fx );
    float4 wy = cubicWeights( fy );
    float wx_xy = (wx.x + wx.y);
    float wx_zw = (wx.z + wx.w);
    float wy_xy = (wy.x + wy.y);
    float wy_zw = (wy.z + wy.w);

    // Get x,y coordinates to sample
    float x0 = (x + wx.y / wx_xy - 0.5f) * invTexWidth;
    float x1 = (x + wx.w / wx_zw + 1.5f) * invTexWidth;
    float y0 = (y + wy.y / wy_xy - 0.5f) * invTexHeight;
    float y1 = (y + wy.w / wy_zw + 1.5f) * invTexHeight;

    // Take 4 texture samples and interpolate between them
    TYPE tex0 = tex2D<TYPE>( context, textureId, x0, y0, isResident );
    TYPE tex1 = tex2D<TYPE>( context, textureId, x1, y0, isResident );
    TYPE tex2 = tex2D<TYPE>( context, textureId, x0, y1, isResident );
    TYPE tex3 = tex2D<TYPE>( context, textureId, x1, y1, isResident );

    float lerpx = wx_zw / (wx_xy + wx_zw);
    float lerpy = wy_zw / (wy_xy + wy_zw);
    return lerp( lerp(tex0, tex1, lerpx), lerp(tex2, tex3, lerpx), lerpy );
}

} // namespace demandLoading