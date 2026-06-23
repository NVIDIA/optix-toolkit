// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

// Tests for the guard against CUDA's 4 GiB mipmapped-array device-sampling limit
// (see src/Util/MipmappedArrayCheck.h).  cuMipmappedArrayCreate succeeds for arrays whose packed
// mip chain exceeds 2^32 bytes, but coarse mip levels of such arrays sample as zero on the device.
// The guard rejects those sizes up front.

#include "Textures/DenseTexture.h"
#include "Textures/SparseTexture.h"
#include "Util/MipmappedArrayCheck.h"
#include "TestDrawTexture.h"

#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <vector>

using namespace demandLoading;
using namespace imageSource;

namespace {

CUDA_ARRAY3D_DESCRIPTOR makeDescriptor( size_t width, size_t height, CUarray_format format, unsigned int numChannels, bool sparse = false )
{
    CUDA_ARRAY3D_DESCRIPTOR ad{};
    ad.Width       = width;
    ad.Height      = height;
    ad.Format      = format;
    ad.NumChannels = numChannels;
    ad.Flags       = sparse ? CUDA_ARRAY3D_SPARSE : 0;
    return ad;
}

// Number of mip levels in the full chain down to 1x1 -- this is what real textures and the sparse
// path (calculateNumMipLevels) use.
unsigned int fullChainLevels( size_t width, size_t height )
{
    return calculateNumMipLevels( static_cast<unsigned int>( width ), static_cast<unsigned int>( height ) );
}

bool supportedFullChain( size_t width, size_t height, CUarray_format format, unsigned int numChannels )
{
    CUDA_ARRAY3D_DESCRIPTOR ad = makeDescriptor( width, height, format, numChannels );
    return isMipmappedArraySizeSupported( ad, fullChainLevels( width, height ) );
}

}  // namespace

//------------------------------------------------------------------------------
// Predicate boundary tests (no allocation).
//
// Each pair straddles the 2^32-byte limit at adjacent dimensions over the full mip chain.  A check
// that only looked at the base level (rather than the whole packed chain) would not flip between
// these neighbors, so these pairs pin down that the guarded quantity is the packed chain size.
//------------------------------------------------------------------------------

TEST( TestMipmappedArraySize, Float4SquareBoundary )
{
    EXPECT_TRUE( supportedFullChain( 14189, 14189, CU_AD_FORMAT_FLOAT, 4 ) );
    EXPECT_FALSE( supportedFullChain( 14190, 14190, CU_AD_FORMAT_FLOAT, 4 ) );
}

TEST( TestMipmappedArraySize, Half4SquareBoundary )
{
    EXPECT_TRUE( supportedFullChain( 20066, 20066, CU_AD_FORMAT_HALF, 4 ) );
    EXPECT_FALSE( supportedFullChain( 20067, 20067, CU_AD_FORMAT_HALF, 4 ) );
}

TEST( TestMipmappedArraySize, LopsidedBoundary )
{
    // 32768 x 6144 float4 packs to exactly 16 bytes under 2^32; one more row of texels exceeds it.
    EXPECT_TRUE( supportedFullChain( 32768, 6144, CU_AD_FORMAT_FLOAT, 4 ) );
    EXPECT_FALSE( supportedFullChain( 32768, 6145, CU_AD_FORMAT_FLOAT, 4 ) );

    // The same limit applies regardless of which dimension is large.
    EXPECT_TRUE( supportedFullChain( 6144, 32768, CU_AD_FORMAT_FLOAT, 4 ) );
    EXPECT_FALSE( supportedFullChain( 6145, 32768, CU_AD_FORMAT_FLOAT, 4 ) );
}

TEST( TestMipmappedArraySize, SingleChannelFormatsHaveHigherLimit )
{
    // 1-channel formats reach the limit at larger dimensions than their 4-channel counterparts,
    // confirming the limit tracks bytes, not texel count. (float: ~28378, half: ~40132.)
    EXPECT_TRUE( supportedFullChain( 28378, 28378, CU_AD_FORMAT_FLOAT, 1 ) );
    EXPECT_FALSE( supportedFullChain( 28379, 28379, CU_AD_FORMAT_FLOAT, 1 ) );
    EXPECT_TRUE( supportedFullChain( 16384, 16384, CU_AD_FORMAT_HALF, 1 ) );  // half1 16384^2 well under 4 GiB
}

// The guard must sum the whole packed chain, not just the base level.  float4 16384^2 has a base
// level of exactly 2^32 bytes (supported on its own), but the full chain exceeds the limit.
TEST( TestMipmappedArraySize, ChainVersusBaseLevel )
{
    CUDA_ARRAY3D_DESCRIPTOR ad = makeDescriptor( 16384, 16384, CU_AD_FORMAT_FLOAT, 4 );

    EXPECT_EQ( MAX_MIPMAPPED_ARRAY_BYTES, getMipmappedArrayPackedBytes( ad, 1 ) );
    EXPECT_TRUE( isMipmappedArraySizeSupported( ad, 1 ) );  // base level only

    EXPECT_GT( getMipmappedArrayPackedBytes( ad, fullChainLevels( 16384, 16384 ) ), MAX_MIPMAPPED_ARRAY_BYTES );
    EXPECT_FALSE( isMipmappedArraySizeSupported( ad, fullChainLevels( 16384, 16384 ) ) );
}

// Three-channel descriptors are counted as four channels, matching imageSource::getBitsPerPixel().
TEST( TestMipmappedArraySize, ThreeChannelsCountedAsFour )
{
    CUDA_ARRAY3D_DESCRIPTOR threeCh = makeDescriptor( 1024, 1024, CU_AD_FORMAT_FLOAT, 3 );
    CUDA_ARRAY3D_DESCRIPTOR fourCh  = makeDescriptor( 1024, 1024, CU_AD_FORMAT_FLOAT, 4 );
    EXPECT_EQ( getMipmappedArrayPackedBytes( fourCh, 1 ), getMipmappedArrayPackedBytes( threeCh, 1 ) );
}

// Sub-byte-per-pixel block-compressed formats must not be truncated to zero bytes per pixel.
// BC1 (4 channels x 1 bit) and BC4 (1 channel x 4 bits) are both 4 bits/pixel.
TEST( TestMipmappedArraySize, BlockCompressedBytesNonZero )
{
    // 4096 x 4096 at 4 bits/pixel = 4096*4096*4/8 = 8,388,608 bytes for the base level.
    const size_t expectedBaseBytes = size_t{ 4096 } * 4096 * 4 / 8;

    CUDA_ARRAY3D_DESCRIPTOR bc1 = makeDescriptor( 4096, 4096, CU_AD_FORMAT_BC1_UNORM, 4 );
    CUDA_ARRAY3D_DESCRIPTOR bc4 = makeDescriptor( 4096, 4096, CU_AD_FORMAT_BC4_UNORM, 1 );

    EXPECT_GT( getMipmappedArrayPackedBytes( bc1, 1 ), size_t{ 0 } );
    EXPECT_GT( getMipmappedArrayPackedBytes( bc4, 1 ), size_t{ 0 } );
    EXPECT_EQ( expectedBaseBytes, getMipmappedArrayPackedBytes( bc1, 1 ) );
    EXPECT_EQ( expectedBaseBytes, getMipmappedArrayPackedBytes( bc4, 1 ) );
}

//------------------------------------------------------------------------------
// Wrapper rejection (no large allocation: the guard throws before cuMipmappedArrayCreate).
//------------------------------------------------------------------------------

TEST( TestMipmappedArraySize, CreateRejectsOversizedDense )
{
    OTK_ERROR_CHECK( cudaSetDevice( 0 ) );
    OTK_ERROR_CHECK( cudaFree( nullptr ) );

    CUDA_ARRAY3D_DESCRIPTOR ad = makeDescriptor( 16384, 16384, CU_AD_FORMAT_FLOAT, 4 );
    CUmipmappedArray        array{};
    EXPECT_THROW( createMipmappedArray( &array, &ad, fullChainLevels( 16384, 16384 ) ), std::runtime_error );
}

TEST( TestMipmappedArraySize, CreateRejectsOversizedSparse )
{
    const unsigned int deviceIndex = getFirstSparseTextureDevice();
    if( deviceIndex == MAX_DEVICES )
        GTEST_SKIP() << "No device supports sparse textures";
    OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
    OTK_ERROR_CHECK( cudaFree( nullptr ) );

    CUDA_ARRAY3D_DESCRIPTOR ad = makeDescriptor( 16384, 16384, CU_AD_FORMAT_FLOAT, 4, /*sparse=*/true );
    CUmipmappedArray        array{};
    EXPECT_THROW( createMipmappedArray( &array, &ad, fullChainLevels( 16384, 16384 ) ), std::runtime_error );
}

//------------------------------------------------------------------------------
// Real DenseTexture / SparseArray init paths.
//------------------------------------------------------------------------------

TEST( TestMipmappedArraySize, DenseTextureInitRejectsOversized )
{
    OTK_ERROR_CHECK( cudaSetDevice( 0 ) );
    OTK_ERROR_CHECK( cudaFree( nullptr ) );

    TextureDescriptor desc{};
    desc.addressMode[0]   = CU_TR_ADDRESS_MODE_CLAMP;
    desc.addressMode[1]   = CU_TR_ADDRESS_MODE_CLAMP;
    desc.filterMode       = CU_TR_FILTER_MODE_POINT;
    desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    desc.maxAnisotropy    = 1;

    TextureInfo info{};
    info.width        = 16384;
    info.height       = 16384;
    info.format       = CU_AD_FORMAT_FLOAT;
    info.numChannels  = 4;
    info.numMipLevels = fullChainLevels( 16384, 16384 );
    info.isValid      = true;

    DenseTexture texture;
    EXPECT_THROW( texture.init( desc, info, nullptr ), std::runtime_error );
}

// The sparse path checks the nominal (full-chain) mip count, not info.numMipLevels.  With
// numMipLevels == 1 a dense 16384^2 float4 texture is fine (base level == 2^32 bytes), but the
// sparse array is created with the full nominal chain and must therefore be rejected.
TEST( TestMipmappedArraySize, SparseUsesNominalMipCount )
{
    const unsigned int deviceIndex = getFirstSparseTextureDevice();
    if( deviceIndex == MAX_DEVICES )
        GTEST_SKIP() << "No device supports sparse textures";
    OTK_ERROR_CHECK( cudaSetDevice( deviceIndex ) );
    OTK_ERROR_CHECK( cudaFree( nullptr ) );

    TextureInfo info{};
    info.width        = 16384;
    info.height       = 16384;
    info.format       = CU_AD_FORMAT_FLOAT;
    info.numChannels  = 4;
    info.numMipLevels = 1;  // caller asks for a single level...
    info.isValid      = true;

    // ...dense honors that and the base level alone fits, so init succeeds.
    {
        TextureDescriptor desc{};
        desc.addressMode[0]   = CU_TR_ADDRESS_MODE_CLAMP;
        desc.addressMode[1]   = CU_TR_ADDRESS_MODE_CLAMP;
        desc.filterMode       = CU_TR_FILTER_MODE_POINT;
        desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
        desc.maxAnisotropy    = 1;
        DenseTexture dense;
        EXPECT_NO_THROW( dense.init( desc, info, nullptr ) );
    }

    // ...but SparseArray creates the full nominal chain, which exceeds the limit, so it must throw.
    {
        SparseArray sparse;
        EXPECT_THROW( sparse.init( info ), std::runtime_error );
    }
}

//------------------------------------------------------------------------------
// Small end-to-end happy path: an in-bounds dense texture still creates and samples correctly.
//------------------------------------------------------------------------------

TEST( TestMipmappedArraySize, SmallDenseTextureSamplesCorrectly )
{
    OTK_ERROR_CHECK( cudaSetDevice( 0 ) );
    OTK_ERROR_CHECK( cudaFree( nullptr ) );

    const unsigned int size = 256;  // ~256 KB base level, well within limits

    TextureDescriptor desc{};
    desc.addressMode[0]   = CU_TR_ADDRESS_MODE_CLAMP;
    desc.addressMode[1]   = CU_TR_ADDRESS_MODE_CLAMP;
    desc.filterMode       = CU_TR_FILTER_MODE_POINT;
    desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    desc.maxAnisotropy    = 1;

    TextureInfo info{};
    info.width        = size;
    info.height       = size;
    info.format       = CU_AD_FORMAT_FLOAT;
    info.numChannels  = 4;
    info.numMipLevels = fullChainLevels( size, size );
    info.isValid      = true;

    DenseTexture texture;
    ASSERT_NO_THROW( texture.init( desc, info, nullptr ) );

    // Fill every mip level with a known color.
    const float4 color = make_float4( 0.25f, 0.5f, 0.75f, 1.0f );
    std::vector<float4> hostData( getTextureSizeInBytes( info ) / sizeof( float4 ), color );

    CUstream stream;
    OTK_ERROR_CHECK( cuStreamCreate( &stream, CU_STREAM_DEFAULT ) );
    texture.fillTexture( stream, reinterpret_cast<const char*>( hostData.data() ), info.width, info.height, /*bufferPinned=*/false );
    OTK_ERROR_CHECK( cuStreamSynchronize( stream ) );

    // Sample the base level at the center of a 2x2 output image.
    float4* devImage;
    OTK_ERROR_CHECK( cudaMalloc( &devImage, 4 * sizeof( float4 ) ) );
    launchDrawTextureKernel( stream, devImage, 2, 2, texture.getTextureObject(),
                             make_float2( 0.f, 0.f ), make_float2( 1.f, 1.f ),
                             make_float2( 0.f, 0.f ), make_float2( 0.f, 0.f ) );
    OTK_ERROR_CHECK( cuStreamSynchronize( stream ) );

    std::vector<float4> result( 4 );
    OTK_ERROR_CHECK( cudaMemcpy( result.data(), devImage, 4 * sizeof( float4 ), cudaMemcpyDeviceToHost ) );

    for( const float4& texel : result )
    {
        EXPECT_FLOAT_EQ( color.x, texel.x );
        EXPECT_FLOAT_EQ( color.y, texel.y );
        EXPECT_FLOAT_EQ( color.z, texel.z );
        EXPECT_FLOAT_EQ( color.w, texel.w );
    }

    OTK_ERROR_CHECK( cudaFree( devImage ) );
    OTK_ERROR_CHECK( cuStreamDestroy( stream ) );
}
