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

#include "SourceDir.h"  // generated from SourceDir.h.in

#include <OptiXToolkit/ImageSource/EXRReader.h>

#ifdef OPTIX_SAMPLE_USE_CORE_EXR
#include <OptiXToolkit/ImageSource/CoreEXRReader.h>
#endif

#include <gtest/gtest.h>

#include <half.h> // Imath

#include <vector_functions.h> // CUDA

#include <algorithm>
#include <cassert>
#include <vector>

using namespace imageSource;

// Some possible half pixel types
struct half2
{
    half x, y;
};

struct half4
{
    half x, y, z, w;
};

// An epsilon equality operators for half pixel types.
bool operator==( const half2& a, const half2& b )
{
    const float epsilon = 0.000001f;
    return ( fabs( a.x - b.x ) < epsilon ) && ( fabs( a.y - b.y ) < epsilon );
}

bool operator==( const half4& a, const half4& b )
{
    const float epsilon = 0.000001f;
    return ( fabs( a.x - b.x ) < epsilon ) && ( fabs( a.y - b.y ) < epsilon ) && ( fabs( a.z - b.z ) < epsilon )
           && ( fabs( a.z - b.z ) < epsilon );
}

// An epsilon equality operator for float pixel types.
bool operator==( const float3& a, const float3& b )
{
    const float epsilon = 0.000001f;
    return ( fabs( a.x - b.x ) < epsilon ) && ( fabs( a.y - b.y ) < epsilon ) && ( fabs( a.z - b.z ) < epsilon );
}

// Get a texel out of an image as float4, and return it as a float3
static float3 getTexel( unsigned int x, unsigned int y, const std::vector<float4>& texels, unsigned int width )
{
    float4 c = texels[y * width + x];
    return make_float3( c.x, c.y, c.z );
}

// float3 output operators for more readable EXCEPT failures.
std::ostream& operator<<( std::ostream& out, const float3& a )
{
    return out << '(' << a.x << ", " << a.y << ", " << a.z << ')';
}

std::ostream& operator<<( std::ostream& out, const half2& a )
{
    return out << '(' << a.x << ", " << a.y << ')';
}

std::ostream& operator<<( std::ostream& out, const half4& a )
{
    return out << '(' << a.x << ", " << a.y << ", " << a.z << "," << a.w << ')';
}

//------------------------------------------------------------------------------
// (empty) Test fixtures
struct TestEXRReader : public testing::Test
{
};

#ifdef OPTIX_SAMPLE_USE_CORE_EXR
struct TestCoreEXRReader : public testing::Test
{
};
#endif

#ifdef OPTIX_SAMPLE_USE_CORE_EXR
// Macro to instantiate the test cases for both Reader typs
#define INSTANTIATE_READER_TESTS( TEST_NAME )                                                                                       \
    TEST_F( TestEXRReader, TEST_NAME ) { run##TEST_NAME<EXRReader>(); }                                                \
    TEST_F( TestCoreEXRReader, TEST_NAME ) { run##TEST_NAME<CoreEXRReader>(); }
#else
// Macro to instantiate the test cases for just the EXRReader
#define INSTANTIATE_READER_TESTS( TEST_NAME )                                                                                       \
    TEST_F( TestEXRReader, TEST_NAME ) { run##TEST_NAME<EXRReader>(); }
#endif

// The test image was constructed with two distinctive miplevels.  The fine miplevel is 128x128,
// with a red/white checkboard pattern (with 16x16 squares), while the coarser miplevel is 64x64,
// with a blue/white checkerboard pattern (with 16x16 squares).  Two versions were created, one
// stored as floats, and another stored as halfs.  They were created as follows:
//   oiiotool --pattern checker:width=16:height=16:color1=1,0,0 128x128 3 -o level0.png
//   oiiotool --pattern checker:width=16:height=16:color1=0,0,1 64x64 3 -o level1.png
//
//   maketx level0.png --mipimage level1.png --tile 32 32 -d float -o TiledMipMappedFloat.exr
//   maketx level0.png --mipimage level1.png --tile 32 32 -o TiledMipMappedHalf.exr

//------------------------------------------------------------------------------
// Tests related to floats

template <class ReaderType>
void runReadInfo()
{
    ReaderType  floatReader( getSourceDir() + "/Textures/TiledMipMappedFloat.exr" );
    TextureInfo floatInfo = {};
    ASSERT_NO_THROW( floatReader.open( &floatInfo ) );

    EXPECT_EQ( 128U, floatInfo.width );
    EXPECT_EQ( 32U, floatReader.getTileWidth() );
    EXPECT_TRUE( floatInfo.numMipLevels >= 2U );  // maketx auto-generates the finer levels.
}

INSTANTIATE_READER_TESTS( ReadInfo );

//------------------------------------------------------------------------------

template <class ReaderType>
void runReadInfoScanline()
{
    ReaderType  floatReader( getSourceDir() + "/Textures/ScanlineFineFloat.exr" );
    TextureInfo floatInfo = {};
    ASSERT_NO_THROW(floatReader.open(&floatInfo));

    EXPECT_EQ(128U, floatInfo.width);
    EXPECT_EQ(128U, floatInfo.height);
    EXPECT_TRUE(floatInfo.numMipLevels == 1U);  // Scanline images are not mip-mapped
}

INSTANTIATE_READER_TESTS(ReadInfoScanline);

//------------------------------------------------------------------------------

template <class ReaderType>
void runReadFineTileFloat()
{
    ReaderType  floatReader( getSourceDir() + "/Textures/TiledMipMappedFloat.exr" );
    TextureInfo floatInfo = {};
    ASSERT_NO_THROW( floatReader.open( &floatInfo ) );

    const unsigned int mipLevel = 0;
    const unsigned int width    = floatReader.getTileWidth();
    const unsigned int height   = floatReader.getTileHeight();

    ASSERT_TRUE( floatInfo.format == CU_AD_FORMAT_FLOAT && floatInfo.numChannels == 4 );
    std::vector<float4> texels( width * height );
    ASSERT_NO_THROW( floatReader.readTile( reinterpret_cast<char*>( texels.data() ), mipLevel, 1, 1, width, height ) );

    // Pattern is red/white checkboard with 16x16 squares
    EXPECT_EQ( make_float3( 1, 0, 0 ), getTexel( 0, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( width - 1, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( 0, height - 1, texels, width ) );
    EXPECT_EQ( make_float3( 1, 0, 0 ), getTexel( width - 1, height - 1, texels, width ) );
}

INSTANTIATE_READER_TESTS( ReadFineTileFloat );

//------------------------------------------------------------------------------

template <class ReaderType>
void runReadFineScanlineFloat()
{
    ReaderType  floatReader( getSourceDir() + "/Textures/ScanlineFineFloat.exr" );
    TextureInfo floatInfo = {};
    ASSERT_NO_THROW( floatReader.open( &floatInfo ) );

    const unsigned int mipLevel = 0;
    const unsigned int width = floatInfo.width;
    const unsigned int height = floatInfo.height;

    ASSERT_TRUE( floatInfo.format == CU_AD_FORMAT_FLOAT && floatInfo.numChannels == 4 );
    std::vector<float4> texels( width * height );
    ASSERT_NO_THROW( floatReader.readMipLevel( reinterpret_cast<char*>( texels.data() ), mipLevel, width, height ) );

    // Pattern is red/white checkboard with 16x16 squares
    EXPECT_EQ( make_float3( 1, 0, 0 ), getTexel( 0, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( width - 1, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( 0, height - 1, texels, width ) );
    EXPECT_EQ( make_float3( 1, 0, 0 ), getTexel( width - 1, height - 1, texels, width ) );
}

INSTANTIATE_READER_TESTS( ReadFineScanlineFloat );

//------------------------------------------------------------------------------

template <class ReaderType>
void runReadCoarseTileFloat()
{
    ReaderType    floatReader( getSourceDir() + "/Textures/TiledMipMappedFloat.exr" );
    TextureInfo   floatInfo = {};
    ASSERT_NO_THROW( floatReader.open( &floatInfo ) );

    const unsigned int mipLevel = 1;
    const unsigned int width    = floatReader.getTileWidth();
    const unsigned int height   = floatReader.getTileHeight();

    ASSERT_TRUE( floatInfo.format == CU_AD_FORMAT_FLOAT && floatInfo.numChannels == 4 );
    std::vector<float4> texels( width * height );
    ASSERT_NO_THROW( floatReader.readTile( reinterpret_cast<char*>( texels.data() ), mipLevel, 0, 0, width, height ) );

    // Pattern is blue/white checkerboard with 16x16 squares.
    EXPECT_EQ( make_float3( 0, 0, 1 ), getTexel( 0, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( width - 1, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( 0, height - 1, texels, width ) );
    EXPECT_EQ( make_float3( 0, 0, 1 ), getTexel( width - 1, height - 1, texels, width ) );
}

//------------------------------------------------------------------------------

template <class ReaderType>
void runReadCoarseScanlineFloat()
{
    ReaderType    floatReader( getSourceDir() + "/Textures/ScanlineCoarseFloat.exr" );
    TextureInfo   floatInfo = {};
    ASSERT_NO_THROW( floatReader.open( &floatInfo ) );

    const unsigned int mipLevel = 0;
    const unsigned int width    = floatInfo.width;
    const unsigned int height   = floatInfo.height;

    ASSERT_TRUE( floatInfo.format == CU_AD_FORMAT_FLOAT && floatInfo.numChannels == 4 );
    std::vector<float4> texels( width * height );
    ASSERT_NO_THROW( floatReader.readMipLevel( reinterpret_cast<char*>( texels.data() ), mipLevel, width, height ) );

    // Pattern is blue/white checkerboard with 16x16 squares.
    EXPECT_EQ( make_float3( 0, 0, 1 ), getTexel( 0, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( width - 1, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( 0, height - 1, texels, width ) );
    EXPECT_EQ( make_float3( 0, 0, 1 ), getTexel( width - 1, height - 1, texels, width ) );
}

INSTANTIATE_READER_TESTS( ReadCoarseScanlineFloat );

//------------------------------------------------------------------------------
// Read a tile that is twice as wide/high as the native tile size.

template <typename ReaderType>
void runReadLargeTile()
{
    ReaderType  floatReader( getSourceDir() + "/Textures/TiledMipMappedFloat.exr" );
    TextureInfo floatInfo = {};
    ASSERT_NO_THROW( floatReader.open( &floatInfo ) );

    const unsigned int mipLevel = 0;
    const unsigned int width    = 2 * floatReader.getTileWidth();
    const unsigned int height   = 2 * floatReader.getTileHeight();

    ASSERT_TRUE( floatInfo.format == CU_AD_FORMAT_FLOAT && floatInfo.numChannels == 4 );
    std::vector<float4> texels( width * height );
    ASSERT_NO_THROW( floatReader.readTile( reinterpret_cast<char*>( texels.data() ), mipLevel, 0, 0, width, height ) );

    // For now we print the texels for visual validation.
    for( unsigned int j = 0; j < height; ++j )
    {
        for( unsigned int i = 0; i < width; ++i )
        {
            float4 texel = texels[j * width + i];
            printf( "%i%i%i ", static_cast<int>( texel.x ), static_cast<int>( texel.y ), static_cast<int>( texel.z ) );
        }
        printf( "\n" );
    }
}

INSTANTIATE_READER_TESTS( ReadLargeTile );

//------------------------------------------------------------------------------
// Tests related to halfs

template <typename ReaderType>
void runReadCoarseTileHalf()
{
    ReaderType  halfReader( getSourceDir() + "/Textures/TiledMipMappedHalf.exr" );
    TextureInfo halfInfo = {};
    ASSERT_NO_THROW( halfReader.open( &halfInfo ) );

    const unsigned int mipLevel = 1;
    const unsigned int width    = halfReader.getTileWidth();
    const unsigned int height   = halfReader.getTileHeight();

    ASSERT_TRUE( halfInfo.format == CU_AD_FORMAT_HALF && halfInfo.numChannels == 4 );
    std::vector<char> buff( width * height * sizeof( half4 ) );
    ASSERT_NO_THROW( halfReader.readTile( buff.data(), mipLevel, 0, 0, width, height ) );
    const half4* texels = reinterpret_cast<const half4*>( buff.data() );

    // Pattern is blue/white checkerboard with 16x16 squares.
    half4 blue{ 0, 0, 1, 0 };
    half4 white{ 1, 1, 1, 0 };
    EXPECT_EQ( blue, texels[0 * width + 0] );
    EXPECT_EQ( white, texels[0 * width + ( width - 1 )] );
    EXPECT_EQ( white, texels[( height - 1 ) * width + 0] );
    EXPECT_EQ( blue, texels[( height - 1 ) * width + ( width - 1 )] );
}

INSTANTIATE_READER_TESTS( ReadCoarseTileHalf );

//------------------------------------------------------------------------------

template <class ReaderType>
void runReadCoarseScanlineHalf()
{
    ReaderType    halfReader( getSourceDir() + "/Textures/ScanlineCoarseHalf.exr" );
    TextureInfo   halfInfo = {};
    ASSERT_NO_THROW( halfReader.open( &halfInfo) );

    const unsigned int mipLevel = 0;
    const unsigned int width    = halfInfo.width;
    const unsigned int height   = halfInfo.height;

    ASSERT_TRUE( halfInfo.format == CU_AD_FORMAT_HALF && halfInfo.numChannels == 4 );
    std::vector<char> buff( width * height * sizeof(half4) );
    ASSERT_NO_THROW( halfReader.readMipLevel( buff.data(), mipLevel, width, height ) );
    const half4* texels = reinterpret_cast<const half4*>( buff.data() );

    // Pattern is blue/white checkerboard with 16x16 squares.
    half4 blue{ 0, 0, 1, 0 };
    half4 white{ 1, 1, 1, 0 };
    EXPECT_EQ( blue, texels[0 * width + 0] );
    EXPECT_EQ( white, texels[0 * width + ( width - 1 )] );
    EXPECT_EQ( white, texels[( height - 1 ) * width + 0] );
    EXPECT_EQ( blue, texels[( height - 1 ) * width + ( width - 1 )] );
}

INSTANTIATE_READER_TESTS( ReadCoarseScanlineHalf );

//------------------------------------------------------------------------------
// Tests for reading non-square images

template <typename ReaderType>
void runReadPartialTileNonSquare()
{
    ReaderType  halfReader( getSourceDir() + "/Textures/TiledMipMappedHalf.exr" );
    TextureInfo halfInfo = {};
    ASSERT_NO_THROW( halfReader.open( &halfInfo ) );

    ReaderType  nonSquareReader( getSourceDir() + "/Textures/TiledMipMapped124x72.exr" );
    TextureInfo nonSquareInfo = {};
    ASSERT_NO_THROW( nonSquareReader.open( &nonSquareInfo ) );

    const unsigned int mipLevel    = 1;
    const unsigned int levelWidth  = nonSquareInfo.width / ( 1 << mipLevel );
    const unsigned int levelHeight = nonSquareInfo.height / ( 1 << mipLevel );
    const unsigned int tileWidth   = nonSquareReader.getTileWidth();
    const unsigned int tileHeight  = nonSquareReader.getTileHeight();

    const unsigned int numTilesX = ( levelWidth + tileWidth - 1 ) / tileWidth;
    const unsigned int numTilesY = ( levelHeight + tileHeight - 1 ) / tileHeight;

    ASSERT_TRUE( halfInfo.format == CU_AD_FORMAT_HALF && halfInfo.numChannels == 4 );
    std::vector<char> buff( tileWidth * tileHeight * sizeof( half4 ) );
    ASSERT_NO_THROW( nonSquareReader.readTile( buff.data(), mipLevel, numTilesX - 1, numTilesY - 1, tileWidth, tileHeight ) );
    const half4* texels = reinterpret_cast<const half4*>( buff.data() );

    // Pattern is blue/white checkerboard with 16x16 squares.
    // Pixels that are off the image should be black
    half4 black{ 0, 0, 0, 0 };
    half4 blue{ 0, 0, 1, 0 };
    EXPECT_EQ( blue, texels[0 * tileWidth + 0] );
    EXPECT_EQ( black, texels[( tileHeight - 1 ) * tileWidth + ( tileWidth - 1 )] );
}

INSTANTIATE_READER_TESTS( ReadPartialTileNonSquare );
