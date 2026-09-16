// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/FourierBsdfTable.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace demandPbrtScene {

namespace {

static_assert( sizeof( int ) == sizeof( std::int32_t ), "Fourier table parser expects 32-bit int storage" );
static_assert( sizeof( float ) == sizeof( std::uint32_t ), "Fourier table parser expects 32-bit float storage" );

constexpr char SCATFUN_HEADER[8] = { 'S', 'C', 'A', 'T', 'F', 'U', 'N', '\x01' };

FourierBsdfTableLoadResult makeFailure( FourierBsdfTableLoadStatus status, const std::string& diagnostic )
{
    FourierBsdfTableLoadResult result{};
    result.status     = status;
    result.diagnostic = diagnostic;
    return result;
}

FourierBsdfTableLoadResult makeSuccess()
{
    FourierBsdfTableLoadResult result{};
    result.status = FourierBsdfTableLoadStatus::SUCCESS;
    return result;
}

std::string sectionDiagnostic( const std::string& fileName, const std::string& section )
{
    return "Truncated Fourier BSDF table \"" + fileName + "\" while reading " + section;
}

bool isLittleEndian()
{
    const std::uint32_t value{ 1U };
    return *reinterpret_cast<const unsigned char*>( &value ) == 1U;
}

std::uint32_t byteSwap32( std::uint32_t value )
{
    return ( ( value & 0x000000ffU ) << 24 ) | ( ( value & 0x0000ff00U ) << 8 ) | ( ( value & 0x00ff0000U ) >> 8 )
           | ( ( value & 0xff000000U ) >> 24 );
}

bool readBytes( std::istream& input, void* target, std::size_t count )
{
    if( count > static_cast<std::size_t>( std::numeric_limits<std::streamsize>::max() ) )
    {
        return false;
    }
    input.read( static_cast<char*>( target ), static_cast<std::streamsize>( count ) );
    return input.good();
}

bool readInt32Vector( std::istream& input, std::vector<int>& target, std::size_t count )
{
    target.resize( count );
    if( count == 0 )
    {
        return true;
    }

    if( !readBytes( input, target.data(), count * sizeof( std::int32_t ) ) )
    {
        return false;
    }
    if( !isLittleEndian() )
    {
        for( int& value : target )
        {
            std::uint32_t bits{};
            std::memcpy( &bits, &value, sizeof( bits ) );
            bits = byteSwap32( bits );
            std::memcpy( &value, &bits, sizeof( value ) );
        }
    }
    return true;
}

bool readFloatVector( std::istream& input, std::vector<float>& target, std::size_t count )
{
    target.resize( count );
    if( count == 0 )
    {
        return true;
    }

    if( !readBytes( input, target.data(), count * sizeof( float ) ) )
    {
        return false;
    }
    if( !isLittleEndian() )
    {
        for( float& value : target )
        {
            std::uint32_t bits{};
            std::memcpy( &bits, &value, sizeof( bits ) );
            bits = byteSwap32( bits );
            std::memcpy( &value, &bits, sizeof( value ) );
        }
    }
    return true;
}

bool readInt32( std::istream& input, int& value )
{
    std::vector<int> values;
    if( !readInt32Vector( input, values, 1U ) )
    {
        return false;
    }
    value = values[0];
    return true;
}

bool readFloat( std::istream& input, float& value )
{
    std::vector<float> values;
    if( !readFloatVector( input, values, 1U ) )
    {
        return false;
    }
    value = values[0];
    return true;
}

bool checkedMultiply( std::size_t lhs, std::size_t rhs, std::size_t& result )
{
    if( lhs != 0U && rhs > std::numeric_limits<std::size_t>::max() / lhs )
    {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool checkedAdd( std::size_t lhs, std::size_t rhs, std::size_t& result )
{
    if( rhs > std::numeric_limits<std::size_t>::max() - lhs )
    {
        return false;
    }
    result = lhs + rhs;
    return true;
}

std::string unsupportedDiagnostic( const std::string& fileName, const FourierBsdfTable& table )
{
    std::ostringstream out;
    out << "Unsupported Fourier BSDF table \"" << fileName << "\": flags=" << table.flags
        << ", nChannels=" << table.nChannels << ", nBases=" << table.nBases;
    return out.str();
}

FourierBsdfTableLoadResult readTableHeader( const std::string& fileName,
                                            std::ifstream&     input,
                                            std::streamoff&    fileSize,
                                            FourierBsdfTable&  table )
{
    input.open( fileName, std::ios::binary | std::ios::ate );
    if( !input )
    {
        return makeFailure( FourierBsdfTableLoadStatus::FILE_NOT_FOUND,
                            "Unable to open Fourier BSDF table file \"" + fileName + "\"" );
    }

    fileSize = input.tellg();
    input.seekg( 0, std::ios::beg );

    char header[8]{};
    if( !readBytes( input, header, sizeof( header ) ) )
    {
        return makeFailure( FourierBsdfTableLoadStatus::TRUNCATED, sectionDiagnostic( fileName, "header" ) );
    }
    if( std::memcmp( header, SCATFUN_HEADER, sizeof( header ) ) != 0 )
    {
        return makeFailure( FourierBsdfTableLoadStatus::INVALID_HEADER,
                            "Invalid Fourier BSDF table header in \"" + fileName + "\"" );
    }

    int unused{};
    if( !readInt32( input, table.flags ) || !readInt32( input, table.nMu ) || !readInt32( input, table.nCoefficients )
        || !readInt32( input, table.maxOrder ) || !readInt32( input, table.nChannels )
        || !readInt32( input, table.nBases ) || !readInt32( input, unused ) || !readInt32( input, unused )
        || !readInt32( input, unused ) || !readFloat( input, table.eta ) || !readInt32( input, unused )
        || !readInt32( input, unused ) || !readInt32( input, unused ) || !readInt32( input, unused ) )
    {
        return makeFailure( FourierBsdfTableLoadStatus::TRUNCATED, sectionDiagnostic( fileName, "metadata" ) );
    }
    return makeSuccess();
}

FourierBsdfTableLoadResult validateMetadata( const std::string& fileName, const FourierBsdfTable& table )
{
    if( table.flags != 1 || ( table.nChannels != 1 && table.nChannels != 3 ) || table.nBases != 1 )
    {
        return makeFailure( FourierBsdfTableLoadStatus::UNSUPPORTED, unsupportedDiagnostic( fileName, table ) );
    }
    if( table.nMu <= 0 || table.nCoefficients < 0 || table.maxOrder <= 0 )
    {
        return makeFailure( FourierBsdfTableLoadStatus::MALFORMED,
                            "Malformed Fourier BSDF table \"" + fileName + "\": invalid dimensions" );
    }
    return makeSuccess();
}

FourierBsdfTableLoadResult computeGridSize( const std::string& fileName,
                                            const FourierBsdfTable& table,
                                            std::size_t&            gridSize )
{
    const std::size_t nMu{ static_cast<std::size_t>( table.nMu ) };
    if( !checkedMultiply( nMu, nMu, gridSize ) )
    {
        return makeFailure( FourierBsdfTableLoadStatus::MALFORMED,
                            "Malformed Fourier BSDF table \"" + fileName + "\": dimension overflow" );
    }
    return makeSuccess();
}

FourierBsdfTableLoadResult readArrays( std::istream&     input,
                                       const std::string& fileName,
                                       std::size_t        gridSize,
                                       FourierBsdfTable&  table,
                                       std::vector<int>&  offsetsAndLengths )
{
    const std::size_t nMu{ static_cast<std::size_t>( table.nMu ) };
    if( !readFloatVector( input, table.mu, nMu ) )
    {
        return makeFailure( FourierBsdfTableLoadStatus::TRUNCATED, sectionDiagnostic( fileName, "mu values" ) );
    }
    if( !readFloatVector( input, table.cdf, gridSize ) )
    {
        return makeFailure( FourierBsdfTableLoadStatus::TRUNCATED, sectionDiagnostic( fileName, "cdf values" ) );
    }

    std::size_t offsetPairCount{};
    if( !checkedMultiply( gridSize, 2U, offsetPairCount ) )
    {
        return makeFailure( FourierBsdfTableLoadStatus::MALFORMED,
                            "Malformed Fourier BSDF table \"" + fileName + "\": offset table overflow" );
    }
    if( !readInt32Vector( input, offsetsAndLengths, offsetPairCount ) )
    {
        return makeFailure( FourierBsdfTableLoadStatus::TRUNCATED,
                            sectionDiagnostic( fileName, "coefficient offsets" ) );
    }
    if( !readFloatVector( input, table.coefficients, static_cast<std::size_t>( table.nCoefficients ) ) )
    {
        return makeFailure( FourierBsdfTableLoadStatus::TRUNCATED, sectionDiagnostic( fileName, "coefficients" ) );
    }
    return makeSuccess();
}

FourierBsdfTableLoadResult validateCoefficientSpans( const std::string&      fileName,
                                                     const std::vector<int>& offsetsAndLengths,
                                                     std::size_t             gridSize,
                                                     FourierBsdfTable&       table )
{
    table.coefficientOffsets.resize( gridSize );
    table.coefficientCounts.resize( gridSize );
    table.zeroOrderCoefficients.resize( gridSize );
    for( std::size_t i = 0; i < gridSize; ++i )
    {
        const int offset{ offsetsAndLengths[2U * i] };
        const int length{ offsetsAndLengths[2U * i + 1U] };
        if( offset < 0 || length < 0 || length > table.maxOrder )
        {
            return makeFailure( FourierBsdfTableLoadStatus::MALFORMED,
                                "Malformed Fourier BSDF table \"" + fileName + "\": invalid coefficient span" );
        }

        std::size_t channelCoefficientCount{};
        if( !checkedMultiply( static_cast<std::size_t>( length ), static_cast<std::size_t>( table.nChannels ),
                              channelCoefficientCount ) )
        {
            return makeFailure( FourierBsdfTableLoadStatus::MALFORMED,
                                "Malformed Fourier BSDF table \"" + fileName + "\": coefficient span overflow" );
        }

        std::size_t spanEnd{};
        if( !checkedAdd( static_cast<std::size_t>( offset ), channelCoefficientCount, spanEnd )
            || spanEnd > table.coefficients.size() )
        {
            return makeFailure( FourierBsdfTableLoadStatus::MALFORMED,
                                "Malformed Fourier BSDF table \"" + fileName
                                    + "\": coefficient span exceeds coefficient data" );
        }

        table.coefficientOffsets[i]    = offset;
        table.coefficientCounts[i]     = length;
        table.zeroOrderCoefficients[i] = length > 0 ? table.coefficients[static_cast<std::size_t>( offset )] : 0.0f;
    }
    return makeSuccess();
}

void recordTrailingByteCount( std::istream& input, std::streamoff fileSize, FourierBsdfTable& table )
{
    const std::streamoff tableEnd{ input.tellg() };
    if( tableEnd >= 0 && fileSize >= tableEnd )
    {
        table.trailingByteCount = static_cast<std::size_t>( fileSize - tableEnd );
    }
}

FourierBsdfTableLoadResult makeSuccess( FourierBsdfTable&& table )
{
    FourierBsdfTableLoadResult result{ makeSuccess() };
    result.table = std::move( table );
    return result;
}

}  // namespace

FourierBsdfTableLoadResult loadFourierBsdfTable( const std::string& fileName )
{
    std::ifstream input;
    std::streamoff fileSize{};
    FourierBsdfTable table{};
    std::size_t gridSize{};
    std::vector<int> offsetsAndLengths;

    if( const FourierBsdfTableLoadResult result{ readTableHeader( fileName, input, fileSize, table ) }; !result )
    {
        return result;
    }
    if( const FourierBsdfTableLoadResult result{ validateMetadata( fileName, table ) }; !result )
    {
        return result;
    }
    if( const FourierBsdfTableLoadResult result{ computeGridSize( fileName, table, gridSize ) }; !result )
    {
        return result;
    }
    if( const FourierBsdfTableLoadResult result{ readArrays( input, fileName, gridSize, table, offsetsAndLengths ) };
        !result )
    {
        return result;
    }
    if( const FourierBsdfTableLoadResult result{
            validateCoefficientSpans( fileName, offsetsAndLengths, gridSize, table ) };
        !result )
    {
        return result;
    }

    recordTrailingByteCount( input, fileSize, table );
    return makeSuccess( std::move( table ) );
}

}  // namespace demandPbrtScene
