// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Config.h"                 // generated from Config.h.in
#include "ImageSourceTestConfig.h"  // generated from ImageSourceTestConfig.h.in

#ifdef OPTIX_SAMPLE_USE_CORE_EXR
#include <OptiXToolkit/ImageSource/CoreEXRReader.h>
#endif
#if OTK_USE_OPENEXR
#include <OptiXToolkit/ImageSource/EXRReader.h>
#endif
#if OTK_USE_OIIO
#include <OptiXToolkit/ImageSource/OIIOReader.h>
#endif
#include <OptiXToolkit/ShaderUtil/vec_printers.h>

#include <gtest/gtest.h>

#include <half.h> // Imath

#include <vector_functions.h> // CUDA

#include <algorithm>
#include <atomic>
#include <cassert>
#include <string>
#include <thread>
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

struct TestOIIOReader : public testing::Test
{
};

// Macro to instantiate the test cases for all Reader types
#if OTK_USE_OPENEXR && OTK_USE_OIIO
#define INSTANTIATE_READER_TESTS( TEST_NAME )                                                                          \
    TEST_F( TestEXRReader, TEST_NAME ) { run##TEST_NAME<EXRReader>(); }                                                \
    TEST_F( TestCoreEXRReader, TEST_NAME ) { run##TEST_NAME<CoreEXRReader>(); }                                        \
    TEST_F( TestOIIOReader, TEST_NAME ) { run##TEST_NAME<OIIOReader>(); }
#elif OTK_USE_OPENEXR
#define INSTANTIATE_READER_TESTS( TEST_NAME )                                                                          \
    TEST_F( TestEXRReader, TEST_NAME ) { run##TEST_NAME<EXRReader>(); }                                                \
    TEST_F( TestCoreEXRReader, TEST_NAME ) { run##TEST_NAME<CoreEXRReader>(); } 
#elif OTK_USE_OIIO
#define INSTANTIATE_READER_TESTS( TEST_NAME )                                                                          \
    TEST_F( TestOIIOReader, TEST_NAME ) { run##TEST_NAME<OIIOReader>(); }
#else
#define INSTANTIATE_READER_TESTS( TEST_NAME )
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

INSTANTIATE_READER_TESTS( ReadInfo )

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

INSTANTIATE_READER_TESTS(ReadInfoScanline)

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
    ASSERT_NO_THROW( floatReader.readTile( reinterpret_cast<char*>( texels.data() ), mipLevel, { 1, 1, width, height }, nullptr ) );

    // Pattern is red/white checkerboard with 16x16 squares
    EXPECT_EQ( make_float3( 1, 0, 0 ), getTexel( 0, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( width - 1, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( 0, height - 1, texels, width ) );
    EXPECT_EQ( make_float3( 1, 0, 0 ), getTexel( width - 1, height - 1, texels, width ) );
}

INSTANTIATE_READER_TESTS( ReadFineTileFloat )

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
    ASSERT_NO_THROW( floatReader.readMipLevel( reinterpret_cast<char*>( texels.data() ), mipLevel, width, height, nullptr ) );

    // Pattern is red/white checkboard with 16x16 squares
    EXPECT_EQ( make_float3( 1, 0, 0 ), getTexel( 0, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( width - 1, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( 0, height - 1, texels, width ) );
    EXPECT_EQ( make_float3( 1, 0, 0 ), getTexel( width - 1, height - 1, texels, width ) );
}

INSTANTIATE_READER_TESTS( ReadFineScanlineFloat )

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
    ASSERT_NO_THROW( floatReader.readTile( reinterpret_cast<char*>( texels.data() ), mipLevel, { 0, 0, width, height }, nullptr ) );

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
    ASSERT_NO_THROW( floatReader.readMipLevel( reinterpret_cast<char*>( texels.data() ), mipLevel, width, height, nullptr ) );

    // Pattern is blue/white checkerboard with 16x16 squares.
    EXPECT_EQ( make_float3( 0, 0, 1 ), getTexel( 0, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( width - 1, 0, texels, width ) );
    EXPECT_EQ( make_float3( 1, 1, 1 ), getTexel( 0, height - 1, texels, width ) );
    EXPECT_EQ( make_float3( 0, 0, 1 ), getTexel( width - 1, height - 1, texels, width ) );
}

INSTANTIATE_READER_TESTS( ReadCoarseScanlineFloat )

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
    ASSERT_NO_THROW( floatReader.readTile( reinterpret_cast<char*>( texels.data() ), mipLevel, { 0, 0, width, height }, nullptr ) );

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

INSTANTIATE_READER_TESTS( ReadLargeTile )

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
    ASSERT_NO_THROW( halfReader.readTile( buff.data(), mipLevel, { 0, 0, width, height }, nullptr ) );
    const half4* texels = reinterpret_cast<const half4*>( buff.data() );

    // Pattern is blue/white checkerboard with 16x16 squares.
    half4 blue{ 0, 0, 1, 0 };
    half4 white{ 1, 1, 1, 0 };
    EXPECT_EQ( blue, texels[0 * width + 0] );
    EXPECT_EQ( white, texels[0 * width + ( width - 1 )] );
    EXPECT_EQ( white, texels[( height - 1 ) * width + 0] );
    EXPECT_EQ( blue, texels[( height - 1 ) * width + ( width - 1 )] );
}

INSTANTIATE_READER_TESTS( ReadCoarseTileHalf )

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
    ASSERT_NO_THROW( halfReader.readMipLevel( buff.data(), mipLevel, width, height, nullptr ) );
    const half4* texels = reinterpret_cast<const half4*>( buff.data() );

    // Pattern is blue/white checkerboard with 16x16 squares.
    half4 blue{ 0, 0, 1, 0 };
    half4 white{ 1, 1, 1, 0 };
    EXPECT_EQ( blue, texels[0 * width + 0] );
    EXPECT_EQ( white, texels[0 * width + ( width - 1 )] );
    EXPECT_EQ( white, texels[( height - 1 ) * width + 0] );
    EXPECT_EQ( blue, texels[( height - 1 ) * width + ( width - 1 )] );
}

INSTANTIATE_READER_TESTS( ReadCoarseScanlineHalf )

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
    ASSERT_NO_THROW( nonSquareReader.readTile( buff.data(), mipLevel,
                                               { numTilesX - 1, numTilesY - 1, tileWidth, tileHeight }, nullptr ) );
    const half4* texels = reinterpret_cast<const half4*>( buff.data() );

    // Pattern is blue/white checkerboard with 16x16 squares.
    // Pixels that are off the image should be black
    half4 black{ 0, 0, 0, 0 };
    half4 blue{ 0, 0, 1, 0 };
    EXPECT_EQ( blue, texels[0 * tileWidth + 0] );

    // Pixels off the right/bottom edge of the level must stay black. OIIOReader previously
    // mis-placed edge tile rows here; the single read_tiles() over the clamped region fixes it.
    EXPECT_EQ( black, texels[( tileHeight - 1 ) * tileWidth + ( tileWidth - 1 )] );
}

INSTANTIATE_READER_TESTS( ReadPartialTileNonSquare )

#if OTK_USE_OIIO

template <typename T>
std::array<T, 3> getTexel( unsigned int x, unsigned int y, const std::vector<std::array<T, 4>>& texels, unsigned int width )
{
    std::array<T, 4> c = texels[y * width + x];
    return std::array<T, 3>{c[0], c[1], c[2]};
}

template <typename T>
void checkTile( OIIOReader* reader, std::array<T,3> white, std::array<T,3> red )
{
    const unsigned int mipLevel = 0;
    unsigned int       width    = reader->getTileWidth();
    unsigned int       height   = reader->getTileHeight();
    width                       = width ? width : 32;
    height                      = height ? height : 32;

    ASSERT_EQ( 4, reader->getInfo().numChannels );
    std::vector<std::array<T, 4>> texels( width * height );
    ASSERT_NO_THROW( reader->readTile( reinterpret_cast<char*>( texels.data() ), mipLevel, { 1, 1, width, height }, nullptr ) );

    // Pattern is red/white checkboard with 16x16 squares
    EXPECT_EQ( red, getTexel( 0, 0, texels, width ) );
    EXPECT_EQ( white, getTexel( width - 1, 0, texels, width ) );
    EXPECT_EQ( white, getTexel( 0, height - 1, texels, width ) );
    EXPECT_EQ( (red), getTexel( width - 1, height - 1, texels, width ) );
}

class TestReaderFileTypes : public testing::TestWithParam<std::string>
{
};

TEST_P( TestReaderFileTypes, ReadTile )
{
    std::string filename( GetParam() );
    OIIOReader  reader( getSourceDir() + "/Textures/" + filename );
    TextureInfo info = {};
    ASSERT_NO_THROW( reader.open( &info ) );
    if( info.format == CU_AD_FORMAT_UNSIGNED_INT8 )
    {
        std::array<uint8_t, 3> white{ 255, 255, 255 };
        std::array<uint8_t, 3> red{ 254, 0, 0 };
        checkTile<uint8_t>( &reader, white, red );
    }
    else
    {
        ASSERT_EQ( CU_AD_FORMAT_FLOAT, info.format );
        std::array<float, 3> white{ 1, 1, 1 };
        std::array<float, 3> red{ 1, 0, 0 };
        checkTile<float>( &reader, white, red );
    }
}

INSTANTIATE_TEST_SUITE_P( OIIO,
                          TestReaderFileTypes,
                          testing::Values( "TiledMipMappedFloat.tif", "level0.png", "level0.jpg" ) );

// readTile is lock-free and thread-safe: many threads reading tiles of the same open reader
// must each get bytes identical to a single-threaded reference read. (Run under ThreadSanitizer
// to also catch data races.)
TEST( TestOIIOReaderThreadSafety, ConcurrentReadTile )
{
    OIIOReader reader( getSourceDir() + "/Textures/TiledMipMappedHalf.exr" );

    TextureInfo info = {};
    ASSERT_NO_THROW( reader.open( &info ) );

    // OIIOReader's constructor requests OIIO's lock-free OpenEXR Core reader.
    EXPECT_EQ( 1, OIIO::get_int_attribute( "openexr:core" ) );

    const unsigned int tw = reader.getTileWidth();
    const unsigned int th = reader.getTileHeight();
    ASSERT_GT( tw, 0u );
    ASSERT_GT( th, 0u );

    const unsigned int mipLevel  = 0;
    const size_t       tileBytes = static_cast<size_t>( tw ) * th * sizeof( half4 );
    const unsigned int numTilesX = ( info.width + tw - 1 ) / tw;
    const unsigned int numTilesY = ( info.height + th - 1 ) / th;
    const unsigned int numTiles  = numTilesX * numTilesY;

    // Single-threaded reference for every tile.
    std::vector<std::vector<char>> reference( numTiles, std::vector<char>( tileBytes, 0 ) );
    for( unsigned int t = 0; t < numTiles; ++t )
        ASSERT_NO_THROW( reader.readTile( reference[t].data(), mipLevel, { t % numTilesX, t / numTilesX, tw, th }, nullptr ) );

    // Hammer the same open reader from many threads; each result must match the reference.
    std::atomic<bool>        ok{ true };
    const unsigned int       numThreads = 8;
    const unsigned int       iterations = 16;
    std::vector<std::thread> threads;
    for( unsigned int thr = 0; thr < numThreads; ++thr )
    {
        threads.emplace_back( [&, thr] {
            std::vector<char> buff( tileBytes, 0 );
            for( unsigned int it = 0; it < iterations && ok.load(); ++it )
            {
                for( unsigned int t = 0; t < numTiles; ++t )
                {
                    const unsigned int idx = ( t + thr ) % numTiles;  // stagger access order per thread
                    std::fill( buff.begin(), buff.end(), 0 );
                    try
                    {
                        reader.readTile( buff.data(), mipLevel, { idx % numTilesX, idx / numTilesX, tw, th }, nullptr );
                    }
                    catch( ... )
                    {
                        ok.store( false );
                        return;
                    }
                    if( buff != reference[idx] )
                    {
                        ok.store( false );
                        return;
                    }
                }
            }
        } );
    }
    for( auto& t : threads )
        t.join();

    EXPECT_TRUE( ok.load() );
}

// readTile must fill out-of-bounds pixels with black itself, not rely on a pre-zeroed buffer.
// Prefill the destination nonzero so a passing black check proves readTile cleared it.
TEST( TestOIIOReaderEdge, PartialTileBlackFill )
{
    OIIOReader  reader( getSourceDir() + "/Textures/TiledMipMapped124x72.exr" );
    TextureInfo info = {};
    ASSERT_NO_THROW( reader.open( &info ) );

    const unsigned int mipLevel   = 1;
    const unsigned int tileWidth  = reader.getTileWidth();
    const unsigned int tileHeight = reader.getTileHeight();
    ASSERT_GT( tileWidth, 0u );
    ASSERT_GT( tileHeight, 0u );

    const unsigned int levelWidth  = info.width / ( 1 << mipLevel );
    const unsigned int levelHeight = info.height / ( 1 << mipLevel );
    const unsigned int numTilesX   = ( levelWidth + tileWidth - 1 ) / tileWidth;
    const unsigned int numTilesY   = ( levelHeight + tileHeight - 1 ) / tileHeight;

    ASSERT_TRUE( info.format == CU_AD_FORMAT_HALF && info.numChannels == 4 );

    std::vector<char> buff( tileWidth * tileHeight * sizeof( half4 ), char( 0xFF ) );
    ASSERT_NO_THROW( reader.readTile( buff.data(), mipLevel,
                                      { numTilesX - 1, numTilesY - 1, tileWidth, tileHeight }, nullptr ) );
    const half4* texels = reinterpret_cast<const half4*>( buff.data() );

    // Pattern is blue/white checkerboard with 16x16 squares; off-image pixels must be black.
    half4 black{ 0, 0, 0, 0 };
    half4 blue{ 0, 0, 1, 0 };
    EXPECT_EQ( blue, texels[0] );                                                   // in-bounds
    EXPECT_EQ( black, texels[( tileHeight - 1 ) * tileWidth + ( tileWidth - 1 )] );  // off-image corner
}

#endif  // OTK_USE_OIIO
