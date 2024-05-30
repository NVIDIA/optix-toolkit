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

#include <optix.h>

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>

#include "SummedAreaTable.h"
#include "Texture.h"
#include "Util/VecMath.h"

#include <limits>
#include <algorithm>

struct CudaTextureAlphaInputFunctorConfig
{
    cudaTextureObject_t inputTexture;
    float               invWidth;
    float               invHeight;
    uint32_t            layers;
    float               opacityCutoff;
    float               transparencyCutoff;
};

struct WrapperIterator
{
    ushort2 value = {};

    using value_type = ushort2;
    using iterator_category = void;
    using difference_type = size_t;
    using pointer = void;
    using reference = const value_type&;

    __device__  WrapperIterator() {};
    __device__  WrapperIterator( unsigned short x, unsigned short y )
        : value( make_ushort2(x,y) )
    {}

    __device__ ushort2 operator*() const
    {
        return value;
    }
};

template<int DIMENSIONS, typename T>
struct Tex2DTy;

template<> struct Tex2DTy<2, float> { typedef float2 value_type; };
template<> struct Tex2DTy<3, float> { typedef float4 value_type; };
template<> struct Tex2DTy<4, float> { typedef float4 value_type; };

template<> struct Tex2DTy<2, char> { typedef char2 value_type; };
template<> struct Tex2DTy<3, char> { typedef char4 value_type; };
template<> struct Tex2DTy<4, char> { typedef char4 value_type; };

template<> struct Tex2DTy<2, unsigned char> { typedef uchar2 value_type; };
template<> struct Tex2DTy<3, unsigned char> { typedef uchar4 value_type; };
template<> struct Tex2DTy<4, unsigned char> { typedef uchar4 value_type; };

template<> struct Tex2DTy<2, short> { typedef short2 value_type; };
template<> struct Tex2DTy<3, short> { typedef short4 value_type; };
template<> struct Tex2DTy<4, short> { typedef short4 value_type; };

template<> struct Tex2DTy<2, unsigned short> { typedef ushort2 value_type; };
template<> struct Tex2DTy<3, unsigned short> { typedef ushort4 value_type; };
template<> struct Tex2DTy<4, unsigned short> { typedef ushort4 value_type; };

template<> struct Tex2DTy<2, int> { typedef int2 value_type; };
template<> struct Tex2DTy<3, int> { typedef int4 value_type; };
template<> struct Tex2DTy<4, int> { typedef int4 value_type; };

template<> struct Tex2DTy<2, unsigned int> { typedef uint2 value_type; };
template<> struct Tex2DTy<3, unsigned int> { typedef uint4 value_type; };
template<> struct Tex2DTy<4, unsigned int> { typedef uint4 value_type; };

template <cuOmmBaking::CudaTextureAlphaMode MODE, typename T>
inline __device__ typename std::enable_if<MODE == cuOmmBaking::CudaTextureAlphaMode::CHANNEL_X, float>::type
loadTex2D( cudaTextureObject_t tex, float tu, float tv ) {
    return ( float )tex2D<T>( tex, tu, tv );
};

template <cuOmmBaking::CudaTextureAlphaMode MODE, typename T>
inline __device__ typename std::enable_if<MODE == cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Y, float>::type
loadTex2D( cudaTextureObject_t tex, float tu, float tv ) {
    return ( float )tex2D<Tex2DTy<2,T>::value_type>( tex, tu, tv ).y;
};

template <cuOmmBaking::CudaTextureAlphaMode MODE, typename T>
inline __device__ typename std::enable_if<MODE == cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Z, float>::type
loadTex2D( cudaTextureObject_t tex, float tu, float tv ) {
    return ( float )tex2D<Tex2DTy<3, T>::value_type>( tex, tu, tv ).z;
};

template <cuOmmBaking::CudaTextureAlphaMode MODE, typename T>
inline __device__ typename std::enable_if<MODE == cuOmmBaking::CudaTextureAlphaMode::CHANNEL_W, float>::type
loadTex2D( cudaTextureObject_t tex, float tu, float tv ) {
    return ( float )tex2D<Tex2DTy<4, T>::value_type>( tex, tu, tv ).w;
};

template <cuOmmBaking::CudaTextureAlphaMode MODE, typename T>
inline __device__ typename std::enable_if<MODE == cuOmmBaking::CudaTextureAlphaMode::RGB_INTENSITY, float>::type
loadTex2D( cudaTextureObject_t tex, float tu, float tv ) {
    typename Tex2DTy<3, T>::value_type v = tex2D<Tex2DTy<3, T>::value_type>( tex, tu, tv );
    return ( float )( v.x + v.y + v.z ) / 3;
};

template <cuOmmBaking::CudaTextureAlphaMode MODE, typename T>
inline __device__ typename std::enable_if<MODE == cuOmmBaking::CudaTextureAlphaMode::CHANNEL_X, float>::type
loadTex2DLayered( cudaTextureObject_t tex, float tu, float tv, uint32_t layer ) {
    return ( float )tex2DLayered<T>( tex, tu, tv, layer );
};

template <cuOmmBaking::CudaTextureAlphaMode MODE, typename T>
inline __device__ typename std::enable_if<MODE == cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Y, float>::type
loadTex2DLayered( cudaTextureObject_t tex, float tu, float tv, uint32_t layer ) {
    return ( float )tex2DLayered<Tex2DTy<2, T>::value_type>( tex, tu, tv, layer ).y;
};

template <cuOmmBaking::CudaTextureAlphaMode MODE, typename T>
inline __device__ typename std::enable_if<MODE == cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Z, float>::type
loadTex2DLayered( cudaTextureObject_t tex, float tu, float tv, uint32_t layer ) {
    return ( float )tex2DLayered<Tex2DTy<3, T>::value_type>( tex, tu, tv, layer ).z;
};

template <cuOmmBaking::CudaTextureAlphaMode MODE, typename T>
inline __device__ typename std::enable_if<MODE == cuOmmBaking::CudaTextureAlphaMode::CHANNEL_W, float>::type
loadTex2DLayered( cudaTextureObject_t tex, float tu, float tv, uint32_t layer ) {
    return ( float )tex2DLayered<Tex2DTy<4, T>::value_type>( tex, tu, tv, layer ).w;
};

template <cuOmmBaking::CudaTextureAlphaMode MODE, typename T>
inline __device__ typename std::enable_if<MODE == cuOmmBaking::CudaTextureAlphaMode::RGB_INTENSITY, float>::type
loadTex2DLayered( cudaTextureObject_t tex, float tu, float tv, uint32_t layer ) {
    typename Tex2DTy<3, T>::value_type v = tex2DLayered<Tex2DTy<3, T>::value_type>( tex, tu, tv, layer );
    return ( float )( v.x + v.y + v.z ) / 3;
};

template<bool IsLayered, bool IsNormalizedCoords, cuOmmBaking::CudaTextureAlphaMode MODE, typename T>
struct CudaTextureAlphaInputFunctor
{
    CudaTextureAlphaInputFunctorConfig config = {};

    __device__ __host__ CudaTextureAlphaInputFunctor(){};
    __device__ __host__ CudaTextureAlphaInputFunctor( CudaTextureAlphaInputFunctorConfig _config )
        : config( _config ){};

    __device__ cuOmmBaking::OpacityState eval( uint32_t x, uint32_t y ) const
    {
        float tu = x + 0.5f;
        float tv = y + 0.5f;

        if( IsNormalizedCoords )
        {
            tu *= config.invWidth;
            tv *= config.invHeight;
        }
        
        if( IsLayered )
        {
            float value = loadTex2DLayered<MODE, T>(config.inputTexture, tu, tv, 0);

            cuOmmBaking::OpacityState state = cuOmmBaking::OpacityState::STATE_UNKNOWN;
            if( value <= config.transparencyCutoff )
                state = cuOmmBaking::OpacityState::STATE_TRANSPARENT;
            else if( value >= config.opacityCutoff )
                state = cuOmmBaking::OpacityState::STATE_OPAQUE;
            else
                return cuOmmBaking::OpacityState::STATE_UNKNOWN;

            for( uint32_t layer = 1; layer < config.layers; layer++ )
            {
                value = loadTex2DLayered<MODE, T>( config.inputTexture, tu, tv, layer );
                if( value <= config.transparencyCutoff )
                {
                    if( state != cuOmmBaking::OpacityState::STATE_TRANSPARENT )
                        return cuOmmBaking::OpacityState::STATE_UNKNOWN;
                }
                else if( value >= config.opacityCutoff )
                {
                    if( state != cuOmmBaking::OpacityState::STATE_OPAQUE )
                        return cuOmmBaking::OpacityState::STATE_UNKNOWN;
                }
                else
                {
                    return cuOmmBaking::OpacityState::STATE_UNKNOWN;
                }
            }

            return state;
        }
        else
        {
            float value = loadTex2D<MODE,T>( config.inputTexture, tu, tv );

            if( value <= config.transparencyCutoff )
                return cuOmmBaking::OpacityState::STATE_TRANSPARENT;
            else if( value >= config.opacityCutoff )
                return cuOmmBaking::OpacityState::STATE_OPAQUE;
        }

        return cuOmmBaking::OpacityState::STATE_UNKNOWN;
    }

    __device__ WrapperIterator operator()( uint32_t x, uint32_t y ) const
    {
        cuOmmBaking::OpacityState state = eval( x, y );

        if( state == cuOmmBaking::OpacityState::STATE_TRANSPARENT )
            return WrapperIterator( 2, 0 );
        else if( state == cuOmmBaking::OpacityState::STATE_OPAQUE )
            return WrapperIterator( 0, 2 );
        return WrapperIterator( 0, 0 );
    }
};

struct StateTextureInputFunctor
{
    struct config_t
    {
        uint32_t        width  = {};
        uint32_t        height = {};
        const uint8_t*  states = {};
        uint32_t        pitchInBits  = {};
    } config;

    __host__ __device__ StateTextureInputFunctor() {};
    __host__ __device__ StateTextureInputFunctor( config_t _config )
        : config( _config ) {};

    __device__ WrapperIterator operator()( uint32_t x, uint32_t y ) const
    {
        if( x >= config.width || y >= config.height )
            return WrapperIterator();

        const uint32_t bidx = x * 2 + y * config.pitchInBits;

        const uint32_t byte  = ( bidx >> 3 );
        const uint32_t shift = ( bidx & 7 );

        const uint8_t mask = config.states[byte];

        const cuOmmBaking::OpacityState state = ( cuOmmBaking::OpacityState )( ( mask >> shift ) & 0x3 );

        if( state == cuOmmBaking::OpacityState::STATE_TRANSPARENT )
            return WrapperIterator( 2, 0 );  // tranparent
        else if( state == cuOmmBaking::OpacityState::STATE_OPAQUE )
            return WrapperIterator( 0, 2 );  // opaque
        else if( state == cuOmmBaking::OpacityState::STATE_RESERVED )
            return WrapperIterator( 1, 1 );  // updatable
        
        return WrapperIterator( 0, 0 );  // unknown
    }
};

template<bool IsLayered, bool IsNormalizedCoords, cuOmmBaking::CudaTextureAlphaMode MODE, typename T>
__host__ cudaError_t launchTransposedSummedAreaTable4( 
    void*               d_temp_storage,
    size_t &            temp_storage_bytes,
    CudaTextureConfig   config,
    cudaTextureObject_t inputTexture,
    uint2*              output, 
    cudaStream_t        stream )
{
    // horizontal line scan needs to fit in unsigned short
    if( 2 * config.width > std::numeric_limits<unsigned short>::max() )
        return cudaErrorInvalidValue;
    // vertical line scan needs to fit in unsigned int
    if( 2 * config.height * config.width > std::numeric_limits<unsigned int>::max() )
        return cudaErrorInvalidValue;

    CudaTextureAlphaInputFunctorConfig kernelConfig = {};
    kernelConfig.transparencyCutoff = config.transparencyCutoff;
    kernelConfig.opacityCutoff = config.opacityCutoff;
    kernelConfig.layers = config.depth;
    kernelConfig.invWidth = 1.f / ( float )config.width;
    kernelConfig.invHeight = 1.f / ( float )config.height;
    kernelConfig.inputTexture = inputTexture;

    struct CudaTextureAlphaInputFunctor<IsLayered, IsNormalizedCoords, MODE, T> input( kernelConfig );
    sat::TransposedSummedAreaTable<decltype( input ), uint2>(
        d_temp_storage,
        temp_storage_bytes,
        config.width, 
        config.height,
        input,
        config.height, output,
        stream );

    return cudaGetLastError();
}

template<bool IsLayered, bool IsNormalizedCoords, typename T>
__host__ cudaError_t launchTransposedSummedAreaTable3(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    CudaTextureConfig   config,
    cudaTextureObject_t inputTexture,
    uint2* output,
    cudaStream_t        stream )
{
    switch( config.alphaMode )
    {
    case cuOmmBaking::CudaTextureAlphaMode::CHANNEL_X:
        return launchTransposedSummedAreaTable4<IsLayered, IsNormalizedCoords, cuOmmBaking::CudaTextureAlphaMode::CHANNEL_X, T>( d_temp_storage, temp_storage_bytes, config, inputTexture, output, stream );
    case cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Y:
        return launchTransposedSummedAreaTable4<IsLayered, IsNormalizedCoords, cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Y, T>( d_temp_storage, temp_storage_bytes, config, inputTexture, output, stream );
    case cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Z:
        return launchTransposedSummedAreaTable4<IsLayered, IsNormalizedCoords, cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Z, T>( d_temp_storage, temp_storage_bytes, config, inputTexture, output, stream );
    case cuOmmBaking::CudaTextureAlphaMode::CHANNEL_W:
        return launchTransposedSummedAreaTable4<IsLayered, IsNormalizedCoords, cuOmmBaking::CudaTextureAlphaMode::CHANNEL_W, T>( d_temp_storage, temp_storage_bytes, config, inputTexture, output, stream );
    case cuOmmBaking::CudaTextureAlphaMode::RGB_INTENSITY:
        return launchTransposedSummedAreaTable4<IsLayered, IsNormalizedCoords, cuOmmBaking::CudaTextureAlphaMode::RGB_INTENSITY, T>( d_temp_storage, temp_storage_bytes, config, inputTexture, output, stream );
    case cuOmmBaking::CudaTextureAlphaMode::DEFAULT:
    case cuOmmBaking::CudaTextureAlphaMode::MAX_NUM:
        return cudaErrorInvalidChannelDescriptor;
    }
    return cudaErrorInvalidChannelDescriptor;
}

template<bool IsNormalizedCoords, typename T>
__host__ cudaError_t launchTransposedSummedAreaTable2(
    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    CudaTextureConfig   config,
    cudaTextureObject_t inputTexture,
    uint2*              output,
    cudaStream_t        stream )
{
    if( config.depth )
        return launchTransposedSummedAreaTable3<true, IsNormalizedCoords, T>( d_temp_storage, temp_storage_bytes, config, inputTexture, output, stream );
    else
        return launchTransposedSummedAreaTable3<false, IsNormalizedCoords, T>( d_temp_storage, temp_storage_bytes, config, inputTexture, output, stream );
}

template<typename T>
__host__ cudaError_t launchTransposedSummedAreaTable1(
    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    CudaTextureConfig   config,
    cudaTextureObject_t inputTexture,
    uint2*              output,
    cudaStream_t        stream )
{
    if( config.texDesc.normalizedCoords )
        return launchTransposedSummedAreaTable2<true, T>( d_temp_storage, temp_storage_bytes, config, inputTexture, output, stream );
    else
        return launchTransposedSummedAreaTable2<false, T>( d_temp_storage, temp_storage_bytes, config, inputTexture, output, stream );
}

cudaError_t launchSummedAreaTable(
    void* d_temp_storage, 
    size_t& temp_storage_bytes, 
    CudaTextureConfig config,
    cudaTextureObject_t inputTexture,
    uint2* outputSat,
    cudaStream_t stream )
{
    int bits = std::max( { config.chanDesc.x, config.chanDesc.y, config.chanDesc.z, config.chanDesc.w } );
    if( config.chanDesc.f == cudaChannelFormatKindFloat )
        return launchTransposedSummedAreaTable1<float>( d_temp_storage, temp_storage_bytes, config, inputTexture, outputSat, stream );
    else if( config.texDesc.readMode == cudaReadModeNormalizedFloat )
        return launchTransposedSummedAreaTable1<float>( d_temp_storage, temp_storage_bytes, config, inputTexture, outputSat, stream );
    else if( config.chanDesc.f == cudaChannelFormatKindUnsigned )
    {
        if( bits <= 8 )
            return launchTransposedSummedAreaTable1<unsigned char>( d_temp_storage, temp_storage_bytes, config, inputTexture, outputSat, stream );
        else if( bits <= 16 )
            return launchTransposedSummedAreaTable1<unsigned short>( d_temp_storage, temp_storage_bytes, config, inputTexture, outputSat, stream );
        else if( bits <= 32 )
            return launchTransposedSummedAreaTable1<unsigned int>( d_temp_storage, temp_storage_bytes, config, inputTexture, outputSat, stream );
    }
    else if( config.chanDesc.f == cudaChannelFormatKindSigned )
    {
        if( bits <= 8 )
            return launchTransposedSummedAreaTable1<char>( d_temp_storage, temp_storage_bytes, config, inputTexture, outputSat, stream );
        else if( bits <= 16 )
            return launchTransposedSummedAreaTable1<short>( d_temp_storage, temp_storage_bytes, config, inputTexture, outputSat, stream );
        else if( bits <= 32 )
            return launchTransposedSummedAreaTable1<int>( d_temp_storage, temp_storage_bytes, config, inputTexture, outputSat, stream );
    }

    return cudaErrorInvalidChannelDescriptor;
}

cudaError_t launchSummedAreaTable(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    StateTextureConfig config,
    const uint8_t* input,
    uint2* outputSat,
    cudaStream_t stream )
{
    // horizontal line scan needs to fit in unsigned short
    if( 2 * config.width > std::numeric_limits<unsigned short>::max() )
        return cudaErrorInvalidValue;
    // vertical line scan needs to fit in unsigned int
    if( 2 * config.height * config.width > std::numeric_limits<unsigned int>::max() )
        return cudaErrorInvalidValue;

    StateTextureInputFunctor::config_t functorConfig;
    functorConfig.width = config.width;
    functorConfig.height = config.height;
    functorConfig.pitchInBits = config.pitchInBits;
    functorConfig.states = input;
    StateTextureInputFunctor functor( functorConfig );

    return sat::TransposedSummedAreaTable<StateTextureInputFunctor, uint2>(
        d_temp_storage,
        temp_storage_bytes,
        config.width,
        config.height,
        functor,
        config.height, outputSat,
        stream );
}
