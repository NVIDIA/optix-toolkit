// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <DemandPbrtScene/FourierBsdfTable.h>

#include <gmock/gmock.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace demandPbrtScene;
using namespace testing;

namespace {

constexpr char SCATFUN_HEADER[8] = { 'S', 'C', 'A', 'T', 'F', 'U', 'N', '\x01' };

std::filesystem::path pbrtReferenceDir()
{
    return std::filesystem::path{ DEMAND_PBRT_SCENE_TEST_SOURCE_DIR } / "pbrt-reference";
}

std::filesystem::path tempFourierTableFile( const std::string& name )
{
    const std::filesystem::path directory{ std::filesystem::temp_directory_path() / "DemandPbrtSceneFourierTable" };
    std::filesystem::create_directories( directory );
    const std::filesystem::path file{ directory / name };
    std::filesystem::remove( file );
    return file;
}

void writeUint32( std::ostream& output, std::uint32_t value )
{
    const unsigned char bytes[] = {
        static_cast<unsigned char>( value & 0xffU ),
        static_cast<unsigned char>( ( value >> 8 ) & 0xffU ),
        static_cast<unsigned char>( ( value >> 16 ) & 0xffU ),
        static_cast<unsigned char>( ( value >> 24 ) & 0xffU ),
    };
    output.write( reinterpret_cast<const char*>( bytes ), sizeof( bytes ) );
}

void writeInt32( std::ostream& output, int value )
{
    writeUint32( output, static_cast<std::uint32_t>( value ) );
}

void writeFloat( std::ostream& output, float value )
{
    std::uint32_t bits{};
    std::memcpy( &bits, &value, sizeof( bits ) );
    writeUint32( output, bits );
}

void writeHeader( std::ostream& output )
{
    output.write( SCATFUN_HEADER, sizeof( SCATFUN_HEADER ) );
}

void writeMetadata( std::ostream& output, int flags, int nMu, int nCoefficients, int maxOrder, int nChannels, int nBases )
{
    writeInt32( output, flags );
    writeInt32( output, nMu );
    writeInt32( output, nCoefficients );
    writeInt32( output, maxOrder );
    writeInt32( output, nChannels );
    writeInt32( output, nBases );
    writeInt32( output, 0 );
    writeInt32( output, 0 );
    writeInt32( output, 0 );
    writeFloat( output, 1.0f );
    writeInt32( output, 0 );
    writeInt32( output, 0 );
    writeInt32( output, 0 );
    writeInt32( output, 0 );
}

void writeMinimalTable( const std::filesystem::path& fileName, int nCoefficients, int nChannels, int coefficientOffset, int coefficientCount )
{
    std::ofstream output{ fileName, std::ios::binary };
    writeHeader( output );
    writeMetadata( output, 1, 1, nCoefficients, 1, nChannels, 1 );
    writeFloat( output, 1.0f );
    writeFloat( output, 1.0f );
    writeInt32( output, coefficientOffset );
    writeInt32( output, coefficientCount );
    for( int i = 0; i < nCoefficients; ++i )
    {
        writeFloat( output, static_cast<float>( i + 1 ) );
    }
}

void writeCoatedCopperOrderShapeTable( const std::filesystem::path& fileName )
{
    constexpr int maxOrder{ 530 };
    constexpr int nMu{ 2 };
    constexpr int nChannels{ 3 };
    constexpr int gridSize{ nMu * nMu };
    constexpr int nCoefficients{ gridSize * nChannels * maxOrder };

    std::ofstream output{ fileName, std::ios::binary };
    writeHeader( output );
    writeMetadata( output, 1, nMu, nCoefficients, maxOrder, nChannels, 1 );
    writeFloat( output, -1.0f );
    writeFloat( output, 1.0f );
    writeFloat( output, 0.0f );
    writeFloat( output, 1.0f );
    writeFloat( output, 0.0f );
    writeFloat( output, 1.0f );
    for( int entry = 0; entry < gridSize; ++entry )
    {
        writeInt32( output, entry * nChannels * maxOrder );
        writeInt32( output, maxOrder );
    }
    for( int i = 0; i < nCoefficients; ++i )
    {
        writeFloat( output, i % maxOrder == 0 ? 1.0f : 0.0f );
    }
}

}  // namespace

TEST( TestFourierBsdfTable, parsesFixtureTableMetadataAndCoefficientLayout )
{
    const std::filesystem::path fixture{ pbrtReferenceDir() / "bsdfs" / "roughgold_alpha_0.2.bsdf" };

    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( fixture.string() ) };

    ASSERT_TRUE( result ) << result.diagnostic;
    const FourierBsdfTable& table{ result.table };
    EXPECT_EQ( 1, table.flags );
    EXPECT_EQ( 58, table.nMu );
    EXPECT_EQ( 41502, table.nCoefficients );
    EXPECT_EQ( 172, table.maxOrder );
    EXPECT_EQ( 3, table.nChannels );
    EXPECT_EQ( 1, table.nBases );
    EXPECT_FLOAT_EQ( 1.0f, table.eta );
    ASSERT_EQ( 58U, table.mu.size() );
    EXPECT_FLOAT_EQ( -1.0f, table.mu.front() );
    EXPECT_NEAR( -0.9976175f, table.mu[1], 1.0e-6f );
    EXPECT_FLOAT_EQ( 1.0f, table.mu.back() );
    EXPECT_EQ( 3364U, table.cdf.size() );
    EXPECT_EQ( 3364U, table.coefficientOffsets.size() );
    EXPECT_EQ( 3364U, table.coefficientCounts.size() );
    EXPECT_EQ( 3364U, table.zeroOrderCoefficients.size() );
    EXPECT_EQ( 41502U, table.coefficients.size() );
    EXPECT_EQ( 682U, table.trailingByteCount );

    const auto firstNonEmpty = std::find_if( table.coefficientCounts.begin(), table.coefficientCounts.end(),
                                             []( int count ) { return count > 0; } );
    ASSERT_NE( table.coefficientCounts.end(), firstNonEmpty );
    const std::size_t firstNonEmptyIndex{ static_cast<std::size_t>( firstNonEmpty - table.coefficientCounts.begin() ) };
    EXPECT_EQ( 1740U, firstNonEmptyIndex );
    EXPECT_EQ( 0, table.coefficientOffsets[firstNonEmptyIndex] );
    EXPECT_EQ( 1, table.coefficientCounts[firstNonEmptyIndex] );
    EXPECT_FLOAT_EQ( table.coefficients[0], table.zeroOrderCoefficients[firstNonEmptyIndex] );
}

TEST( TestFourierBsdfTable, parsesCoatedCopperOrderShape )
{
    const std::filesystem::path fileName{ tempFourierTableFile( "coated-copper-order.bsdf" ) };
    writeCoatedCopperOrderShapeTable( fileName );

    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( fileName.string() ) };

    ASSERT_TRUE( result ) << result.diagnostic;
    const FourierBsdfTable& table{ result.table };
    EXPECT_EQ( 2, table.nMu );
    EXPECT_EQ( 530, table.maxOrder );
    EXPECT_EQ( 3, table.nChannels );
    EXPECT_EQ( 6360, table.nCoefficients );
    EXPECT_EQ( 4U, table.coefficientOffsets.size() );
    EXPECT_THAT( table.coefficientCounts, Each( 530 ) );
    EXPECT_THAT( table.zeroOrderCoefficients, Each( 1.0f ) );
}

TEST( TestFourierBsdfTable, reportsMissingTable )
{
    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( tempFourierTableFile( "missing.bsdf" ).string() ) };

    EXPECT_EQ( FourierBsdfTableLoadStatus::FILE_NOT_FOUND, result.status );
    EXPECT_THAT( result.diagnostic, HasSubstr( "Unable to open Fourier BSDF table file" ) );
}

TEST( TestFourierBsdfTable, rejectsInvalidHeader )
{
    const std::filesystem::path fileName{ tempFourierTableFile( "invalid-header.bsdf" ) };
    {
        std::ofstream output{ fileName, std::ios::binary };
        output.write( "NOTBSDF!", 8 );
    }

    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( fileName.string() ) };

    EXPECT_EQ( FourierBsdfTableLoadStatus::INVALID_HEADER, result.status );
    EXPECT_THAT( result.diagnostic, HasSubstr( "Invalid Fourier BSDF table header" ) );
}

TEST( TestFourierBsdfTable, rejectsTruncatedTable )
{
    const std::filesystem::path fileName{ tempFourierTableFile( "truncated.bsdf" ) };
    {
        std::ofstream output{ fileName, std::ios::binary };
        writeHeader( output );
        writeInt32( output, 1 );
    }

    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( fileName.string() ) };

    EXPECT_EQ( FourierBsdfTableLoadStatus::TRUNCATED, result.status );
    EXPECT_THAT( result.diagnostic, HasSubstr( "while reading metadata" ) );
}

TEST( TestFourierBsdfTable, rejectsUnsupportedMetadata )
{
    const std::filesystem::path fileName{ tempFourierTableFile( "unsupported.bsdf" ) };
    {
        std::ofstream output{ fileName, std::ios::binary };
        writeHeader( output );
        writeMetadata( output, 1, 1, 1, 1, 2, 1 );
    }

    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( fileName.string() ) };

    EXPECT_EQ( FourierBsdfTableLoadStatus::UNSUPPORTED, result.status );
    EXPECT_THAT( result.diagnostic, HasSubstr( "nChannels=2" ) );
}

TEST( TestFourierBsdfTable, rejectsMalformedCoefficientSpans )
{
    const std::filesystem::path fileName{ tempFourierTableFile( "malformed-span.bsdf" ) };
    writeMinimalTable( fileName, 1, 3, 0, 1 );

    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( fileName.string() ) };

    EXPECT_EQ( FourierBsdfTableLoadStatus::MALFORMED, result.status );
    EXPECT_THAT( result.diagnostic, HasSubstr( "coefficient span exceeds coefficient data" ) );
}
