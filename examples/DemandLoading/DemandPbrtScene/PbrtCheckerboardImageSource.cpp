// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/PbrtCheckerboardImageSource.h"

#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <vector_functions.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace demandPbrtScene {

namespace {

constexpr const char* CHECKERBOARD_KEY_PREFIX{ "pbrt-checkerboard:v1:" };

std::string floatToString( float value )
{
    std::ostringstream str;
    str << std::setprecision( 9 ) << value;
    return str.str();
}

std::string colorToString( const float4& color )
{
    return floatToString( color.x ) + "," + floatToString( color.y ) + "," + floatToString( color.z ) + ","
           + floatToString( color.w );
}

std::vector<std::string> split( const std::string& text, char delimiter )
{
    std::vector<std::string> result;
    std::istringstream       str{ text };
    std::string              part;
    while( std::getline( str, part, delimiter ) )
    {
        result.push_back( part );
    }
    return result;
}

float parseFloat( const std::string& text )
{
    std::size_t pos{};
    const float result{ std::stof( text, &pos ) };
    if( pos != text.size() )
    {
        throw std::invalid_argument( "Invalid PBRT checkerboard float '" + text + "'" );
    }
    return result;
}

float4 parseColor( const std::string& text )
{
    const std::vector<std::string> parts{ split( text, ',' ) };
    if( parts.size() != 4U )
    {
        throw std::invalid_argument( "Invalid PBRT checkerboard color '" + text + "'" );
    }
    return make_float4( parseFloat( parts[0] ),
                        parseFloat( parts[1] ),
                        parseFloat( parts[2] ),
                        parseFloat( parts[3] ) );
}

float4 spectrumParam( const pbrt::ParamSet& params, const std::string& name, const float4& defaultValue )
{
    const pbrt::Spectrum defaultSpectrum{ defaultValue.x };
    const pbrt::Spectrum spectrum{ params.FindOneSpectrum( name, defaultSpectrum ) };
    float                rgb[3]{};
    spectrum.ToRGB( rgb );
    return make_float4( rgb[0], rgb[1], rgb[2], 1.0f );
}

float4 floatParam( const pbrt::ParamSet& params, const std::string& name, float defaultValue )
{
    const float value{ params.FindOneFloat( name, defaultValue ) };
    return make_float4( value, value, value, 1.0f );
}

}  // namespace

std::string makePbrtCheckerboardTextureKey( const PbrtCheckerboardDefinition& definition )
{
    return std::string{ CHECKERBOARD_KEY_PREFIX } + colorToString( definition.tex1 ) + ":"
           + colorToString( definition.tex2 ) + ":" + floatToString( definition.uscale ) + ":"
           + floatToString( definition.vscale ) + ":" + floatToString( definition.udelta ) + ":"
           + floatToString( definition.vdelta );
}

bool isPbrtCheckerboardTextureKey( const std::string& key )
{
    return key.compare( 0, std::char_traits<char>::length( CHECKERBOARD_KEY_PREFIX ), CHECKERBOARD_KEY_PREFIX ) == 0;
}

PbrtCheckerboardDefinition parsePbrtCheckerboardTextureKey( const std::string& key )
{
    if( !isPbrtCheckerboardTextureKey( key ) )
    {
        throw std::invalid_argument( "Not a PBRT checkerboard texture key: " + key );
    }

    const std::vector<std::string> parts{
        split( key.substr( std::char_traits<char>::length( CHECKERBOARD_KEY_PREFIX ) ), ':' ) };
    if( parts.size() != 6U )
    {
        throw std::invalid_argument( "Invalid PBRT checkerboard texture key: " + key );
    }
    return { parseColor( parts[0] ), parseColor( parts[1] ), parseFloat( parts[2] ),
             parseFloat( parts[3] ), parseFloat( parts[4] ), parseFloat( parts[5] ) };
}

std::optional<PbrtCheckerboardDefinition> pbrtCheckerboardDefinition( const otk::pbrt::PbrtTexture& texture )
{
    if( texture.type != "checkerboard" )
    {
        return {};
    }
    if( texture.params.FindOneString( "dimension", "2d" ) != "2d" )
    {
        return {};
    }
    if( texture.params.FindOneString( "mapping", "uv" ) != "uv" )
    {
        return {};
    }
    if( !texture.params.FindTexture( "tex1" ).empty() || !texture.params.FindTexture( "tex2" ).empty() )
    {
        return {};
    }

    PbrtCheckerboardDefinition result{};
    const float4               one{ make_float4( 1.0f, 1.0f, 1.0f, 1.0f ) };
    const float4               zero{ make_float4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    if( texture.valueType == "float" )
    {
        result.tex1 = floatParam( texture.params, "tex1", 1.0f );
        result.tex2 = floatParam( texture.params, "tex2", 0.0f );
    }
    else if( texture.valueType == "color" || texture.valueType == "spectrum" )
    {
        result.tex1 = spectrumParam( texture.params, "tex1", one );
        result.tex2 = spectrumParam( texture.params, "tex2", zero );
    }
    else
    {
        return {};
    }

    result.uscale = texture.params.FindOneFloat( "uscale", 1.0f );
    result.vscale = texture.params.FindOneFloat( "vscale", 1.0f );
    result.udelta = texture.params.FindOneFloat( "udelta", 0.0f );
    result.vdelta = texture.params.FindOneFloat( "vdelta", 0.0f );
    return result;
}

std::string pbrtCheckerboardTextureKey( const otk::pbrt::PbrtTexture& texture )
{
    const std::optional<PbrtCheckerboardDefinition> definition{ pbrtCheckerboardDefinition( texture ) };
    return definition ? makePbrtCheckerboardTextureKey( *definition ) : std::string{};
}

std::shared_ptr<imageSource::ImageSource> createPbrtCheckerboardImageSource(
    const PbrtCheckerboardDefinition& definition )
{
    return std::make_shared<PbrtCheckerboardImageSource>( definition );
}

std::shared_ptr<imageSource::ImageSource> createPbrtCheckerboardImageSource( const std::string& key )
{
    return createPbrtCheckerboardImageSource( parsePbrtCheckerboardTextureKey( key ) );
}

PbrtCheckerboardImageSource::PbrtCheckerboardImageSource( PbrtCheckerboardDefinition definition )
    : m_definition( definition )
{
    m_info.width        = PBRT_CHECKERBOARD_TEXTURE_SIZE;
    m_info.height       = PBRT_CHECKERBOARD_TEXTURE_SIZE;
    m_info.format       = CU_AD_FORMAT_FLOAT;
    m_info.numChannels  = 4;
    m_info.numMipLevels = imageSource::calculateNumMipLevels( m_info.width, m_info.height );
    m_info.isValid      = true;
    m_info.isTiled      = true;
}

void PbrtCheckerboardImageSource::open( imageSource::TextureInfo* info )
{
    if( info != nullptr )
    {
        *info = m_info;
    }
}

float4 PbrtCheckerboardImageSource::pixel( unsigned int x, unsigned int y, unsigned int width, unsigned int height )
    const
{
    const float u{ ( static_cast<float>( x ) + 0.5f ) / static_cast<float>( width ) };
    const float v{ ( static_cast<float>( y ) + 0.5f ) / static_cast<float>( height ) };
    const int   checkerU{ static_cast<int>( std::floor( u * m_definition.uscale + m_definition.udelta ) ) };
    const int   checkerV{ static_cast<int>( std::floor( v * m_definition.vscale + m_definition.vdelta ) ) };
    return ( ( checkerU + checkerV ) % 2 == 0 ) ? m_definition.tex1 : m_definition.tex2;
}

bool PbrtCheckerboardImageSource::readTile( char*              dest,
                                            unsigned int       mipLevel,
                                            const imageSource::Tile& tile,
                                            CUstream /*stream*/ )
{
    if( mipLevel >= m_info.numMipLevels )
    {
        throw std::runtime_error( "Attempt to read from non-existent PBRT checkerboard mip level" );
    }

    const unsigned int levelWidth{ std::max( 1U, m_info.width >> mipLevel ) };
    const unsigned int levelHeight{ std::max( 1U, m_info.height >> mipLevel ) };
    const imageSource::PixelPosition start{ imageSource::pixelPosition( tile ) };
    const unsigned int rowPitch{ ( tile.width * imageSource::getBitsPerPixel( m_info ) )
                                 / imageSource::BITS_PER_BYTE };
    for( unsigned int destY = 0; destY < tile.height; ++destY )
    {
        float4* row = reinterpret_cast<float4*>( dest + destY * rowPitch );
        for( unsigned int destX = 0; destX < tile.width; ++destX )
        {
            const unsigned int x{ start.x + destX };
            const unsigned int y{ start.y + destY };
            row[destX] = ( x < levelWidth && y < levelHeight ) ? pixel( x, y, levelWidth, levelHeight )
                                                               : make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
        }
    }
    return true;
}

bool PbrtCheckerboardImageSource::readMipLevel( char*        dest,
                                                unsigned int mipLevel,
                                                unsigned int width,
                                                unsigned int height,
                                                CUstream /*stream*/ )
{
    if( mipLevel >= m_info.numMipLevels )
    {
        throw std::runtime_error( "Attempt to read from non-existent PBRT checkerboard mip level" );
    }

    float4* pixels = reinterpret_cast<float4*>( dest );
    for( unsigned int y = 0; y < height; ++y )
    {
        float4* row = pixels + y * width;
        for( unsigned int x = 0; x < width; ++x )
        {
            row[x] = pixel( x, y, width, height );
        }
    }
    return true;
}

bool PbrtCheckerboardImageSource::readBaseColor( float4& dest )
{
    dest = make_float4( ( m_definition.tex1.x + m_definition.tex2.x ) * 0.5f,
                        ( m_definition.tex1.y + m_definition.tex2.y ) * 0.5f,
                        ( m_definition.tex1.z + m_definition.tex2.z ) * 0.5f,
                        ( m_definition.tex1.w + m_definition.tex2.w ) * 0.5f );
    return true;
}

}  // namespace demandPbrtScene
