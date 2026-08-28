// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/InverseSrgbImageSource.h>

#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace imageSource {

namespace {

float inverseSrgb( float value )
{
    return value <= 0.04045f ? value / 12.92f : std::pow( ( value + 0.055f ) / 1.055f, 2.4f );
}

float normalizationScale( CUarray_format format )
{
    switch( format )
    {
        case CU_AD_FORMAT_UNSIGNED_INT8:
            return 1.0f / static_cast<float>( std::numeric_limits<std::uint8_t>::max() );
        case CU_AD_FORMAT_UNSIGNED_INT16:
            return 1.0f / static_cast<float>( std::numeric_limits<std::uint16_t>::max() );
        case CU_AD_FORMAT_UNSIGNED_INT32:
            return 1.0f / static_cast<float>( std::numeric_limits<std::uint32_t>::max() );
        case CU_AD_FORMAT_FLOAT:
            return 1.0f;
        default:
            throw std::runtime_error( "Unsupported pixel format for inverse sRGB conversion" );
    }
}

float normalizedValue( const char* source, CUarray_format format )
{
    switch( format )
    {
        case CU_AD_FORMAT_UNSIGNED_INT8:
            return static_cast<float>( *reinterpret_cast<const std::uint8_t*>( source ) ) * normalizationScale( format );
        case CU_AD_FORMAT_UNSIGNED_INT16:
            return static_cast<float>( *reinterpret_cast<const std::uint16_t*>( source ) ) * normalizationScale( format );
        case CU_AD_FORMAT_UNSIGNED_INT32:
            return static_cast<float>( *reinterpret_cast<const std::uint32_t*>( source ) ) * normalizationScale( format );
        case CU_AD_FORMAT_FLOAT:
            return *reinterpret_cast<const float*>( source );
        default:
            throw std::runtime_error( "Unsupported pixel format for inverse sRGB conversion" );
    }
}

unsigned int colorChannelCount( unsigned int numChannels )
{
    return numChannels == 2 ? 1U : std::min( numChannels, 3U );
}

}  // namespace

InverseSrgbImageSource::InverseSrgbImageSource( std::shared_ptr<ImageSource> imageSource )
    : WrappedImageSource( std::move( imageSource ) )
{
    if( WrappedImageSource::isOpen() )
    {
        getBaseInfo();
    }
}

void InverseSrgbImageSource::getBaseInfo()
{
    if( WrappedImageSource::getFillType() != CU_MEMORYTYPE_HOST )
    {
        throw std::runtime_error( "InverseSrgbImageSource requires a host-filled image source" );
    }

    m_baseInfo = WrappedImageSource::getInfo();
    normalizationScale( m_baseInfo.format );
    m_info        = m_baseInfo;
    m_info.format = CU_AD_FORMAT_FLOAT;
}

void InverseSrgbImageSource::open( TextureInfo* info )
{
    if( !WrappedImageSource::isOpen() )
    {
        WrappedImageSource::open( nullptr );
    }
    getBaseInfo();
    if( info != nullptr )
    {
        *info = m_info;
    }
}

void InverseSrgbImageSource::close()
{
    WrappedImageSource::close();
    m_baseInfo = TextureInfo{};
    m_info     = TextureInfo{};
}

const TextureInfo& InverseSrgbImageSource::getInfo() const
{
    return m_info;
}

void InverseSrgbImageSource::convertPixels( const char* source, float* dest, size_t numPixels ) const
{
    const unsigned int bytesPerChannel{ getBitsPerChannel( m_baseInfo.format ) / BITS_PER_BYTE };
    const unsigned int numColorChannels{ colorChannelCount( m_baseInfo.numChannels ) };
    for( size_t pixel = 0; pixel < numPixels; ++pixel )
    {
        for( unsigned int channel = 0; channel < m_baseInfo.numChannels; ++channel )
        {
            const float value{ normalizedValue( source, m_baseInfo.format ) };
            *dest++ = channel < numColorChannels ? inverseSrgb( value ) : value;
            source += bytesPerChannel;
        }
    }
}

bool InverseSrgbImageSource::readTile( char* dest, unsigned int mipLevel, const Tile& tile, CUstream stream )
{
    const size_t baseSize{ static_cast<size_t>( tile.width ) * tile.height * getBitsPerPixel( m_baseInfo ) / BITS_PER_BYTE };
    std::vector<char> basePixels( baseSize );
    if( !WrappedImageSource::readTile( basePixels.data(), mipLevel, tile, stream ) )
    {
        return false;
    }
    convertPixels( basePixels.data(), reinterpret_cast<float*>( dest ), static_cast<size_t>( tile.width ) * tile.height );
    return true;
}

bool InverseSrgbImageSource::readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream )
{
    const size_t      numPixels{ static_cast<size_t>( expectedWidth ) * expectedHeight };
    const size_t      baseSize{ numPixels * getBitsPerPixel( m_baseInfo ) / BITS_PER_BYTE };
    std::vector<char> basePixels( baseSize );
    if( !WrappedImageSource::readMipLevel( basePixels.data(), mipLevel, expectedWidth, expectedHeight, stream ) )
    {
        return false;
    }
    convertPixels( basePixels.data(), reinterpret_cast<float*>( dest ), numPixels );
    return true;
}

bool InverseSrgbImageSource::readMipTail( char* dest, unsigned int mipTailFirstLevel, unsigned int numMipLevels, const uint2* mipLevelDims, CUstream stream )
{
    size_t offset{};
    for( unsigned int mipLevel = mipTailFirstLevel; mipLevel < numMipLevels; ++mipLevel )
    {
        const uint2 levelDims{ mipLevelDims[mipLevel] };
        if( !readMipLevel( dest + offset, mipLevel, levelDims.x, levelDims.y, stream ) )
        {
            return false;
        }
        offset += static_cast<size_t>( levelDims.x ) * levelDims.y * getBitsPerPixel( m_info ) / BITS_PER_BYTE;
    }
    return true;
}

bool InverseSrgbImageSource::readBaseColor( float4& dest )
{
    float4 baseColor{};
    if( !WrappedImageSource::readBaseColor( baseColor ) )
    {
        return false;
    }

    const unsigned int numColorChannels{ colorChannelCount( m_baseInfo.numChannels ) };
    const float        scale{ normalizationScale( m_baseInfo.format ) };
    float*             channels{ &baseColor.x };
    for( unsigned int channel = 0; channel < m_baseInfo.numChannels; ++channel )
    {
        const float value{ channels[channel] * scale };
        channels[channel] = channel < numColorChannels ? inverseSrgb( value ) : value;
    }
    dest = baseColor;
    return true;
}

std::shared_ptr<ImageSource> createInverseSrgbImageSource( std::shared_ptr<ImageSource> imageSource )
{
    if( !imageSource )
    {
        return {};
    }
    return std::make_shared<InverseSrgbImageSource>( std::move( imageSource ) );
}

}  // namespace imageSource
