// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <DemandPbrtScene/FourierBsdfTable.h>

#include <DemandPbrtScene/Testing/FourierBsdfTableWriter.h>

#include <gmock/gmock.h>

#include <algorithm>
#include <filesystem>
#include <fstream>

using demandPbrtScene::testing::FourierBsdfTableWriter;

namespace {

using namespace demandPbrtScene;

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

void writeMalformedLargeSpanTable( const std::filesystem::path& fileName )
{
    constexpr int maxOrder{ 1599 };
    constexpr int nMu{ 2 };
    constexpr int nChannels{ 3 };
    constexpr int gridSize{ nMu * nMu };

    FourierBsdfTableWriter output{ fileName };
    output.writeMetadata( 1, nMu, maxOrder, maxOrder, nChannels, 1 );
    output.writeFloat( -1.0f );
    output.writeFloat( 1.0f );
    output.writeFloat( 0.0f );
    output.writeFloat( 1.0f );
    output.writeFloat( 0.0f );
    output.writeFloat( 1.0f );
    for( int i = 0; i < gridSize; ++i )
    {
        output.writeInt32( 0 );
        output.writeInt32( maxOrder );
    }
    for( int i = 0; i < maxOrder; ++i )
    {
        output.writeFloat( 1.0f );
    }
}

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
    FourierBsdfTableWriter::writeOrderShapeTable( fileName, 530 );

    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( fileName.string() ) };

    ASSERT_TRUE( result ) << result.diagnostic;
    const FourierBsdfTable& table{ result.table };
    EXPECT_EQ( 2, table.nMu );
    EXPECT_EQ( 530, table.maxOrder );
    EXPECT_EQ( 3, table.nChannels );
    EXPECT_EQ( 6360, table.nCoefficients );
    EXPECT_EQ( 4U, table.coefficientOffsets.size() );
    for( int count : table.coefficientCounts )
    {
        EXPECT_EQ( 530, count );
    }
    for( float coefficient : table.zeroOrderCoefficients )
    {
        EXPECT_FLOAT_EQ( 1.0f, coefficient );
    }
}

TEST( TestFourierBsdfTable, parsesCeramicOrderShape )
{
    const std::filesystem::path fileName{ tempFourierTableFile( "ceramic-order.bsdf" ) };
    FourierBsdfTableWriter::writeOrderShapeTable( fileName, 1599 );

    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( fileName.string() ) };

    ASSERT_TRUE( result ) << result.diagnostic;
    const FourierBsdfTable& table{ result.table };
    EXPECT_EQ( 2, table.nMu );
    EXPECT_EQ( 1599, table.maxOrder );
    EXPECT_EQ( 3, table.nChannels );
    EXPECT_EQ( 19188, table.nCoefficients );
    EXPECT_EQ( 4U, table.coefficientOffsets.size() );
    for( int count : table.coefficientCounts )
    {
        EXPECT_EQ( 1599, count );
    }
    for( float coefficient : table.zeroOrderCoefficients )
    {
        EXPECT_FLOAT_EQ( 1.0f, coefficient );
    }
}

TEST( TestFourierBsdfTable, reportsMissingTable )
{
    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( tempFourierTableFile( "missing.bsdf" ).string() ) };

    EXPECT_EQ( FourierBsdfTableLoadStatus::FILE_NOT_FOUND, result.status );
    EXPECT_THAT( result.diagnostic, ::testing::HasSubstr( "Unable to open Fourier BSDF table file" ) );
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
    EXPECT_THAT( result.diagnostic, ::testing::HasSubstr( "Invalid Fourier BSDF table header" ) );
}

TEST( TestFourierBsdfTable, rejectsTruncatedTable )
{
    const std::filesystem::path fileName{ tempFourierTableFile( "truncated.bsdf" ) };
    {
        FourierBsdfTableWriter writer{ fileName };
        writer.writeInt32( 1 );
    }

    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( fileName.string() ) };

    EXPECT_EQ( FourierBsdfTableLoadStatus::TRUNCATED, result.status );
    EXPECT_THAT( result.diagnostic, ::testing::HasSubstr( "while reading metadata" ) );
}

TEST( TestFourierBsdfTable, rejectsUnsupportedMetadata )
{
    const std::filesystem::path fileName{ tempFourierTableFile( "unsupported.bsdf" ) };
    {
        FourierBsdfTableWriter writer{ fileName };
        writer.writeMetadata( 1, 1, 1, 1, 2, 1 );
    }

    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( fileName.string() ) };

    EXPECT_EQ( FourierBsdfTableLoadStatus::UNSUPPORTED, result.status );
    EXPECT_THAT( result.diagnostic, ::testing::HasSubstr( "nChannels=2" ) );
}

TEST( TestFourierBsdfTable, rejectsMalformedCoefficientSpans )
{
    const std::filesystem::path fileName{ tempFourierTableFile( "malformed-span.bsdf" ) };
    FourierBsdfTableWriter::writeMinimalTable( fileName, { 1.0f }, 3, 0, 1 );

    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( fileName.string() ) };

    EXPECT_EQ( FourierBsdfTableLoadStatus::MALFORMED, result.status );
    EXPECT_THAT( result.diagnostic, ::testing::HasSubstr( "coefficient span exceeds coefficient data" ) );
}

}  // namespace

TEST( TestFourierBsdfTable, rejectsMalformedLargeCoefficientSpans )
{
    const std::filesystem::path fileName{ tempFourierTableFile( "malformed-large-span.bsdf" ) };
    writeMalformedLargeSpanTable( fileName );

    const FourierBsdfTableLoadResult result{ loadFourierBsdfTable( fileName.string() ) };

    EXPECT_EQ( FourierBsdfTableLoadStatus::MALFORMED, result.status );
    EXPECT_THAT( result.diagnostic, ::testing::HasSubstr( "coefficient span exceeds coefficient data" ) );
}
