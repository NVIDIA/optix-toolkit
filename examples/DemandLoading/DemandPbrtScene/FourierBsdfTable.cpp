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

}  // namespace

FourierBsdfTableLoadResult loadFourierBsdfTable( const std::string& fileName )
{
    std::ifstream input{ fileName, std::ios::binary | std::ios::ate };
    if( !input )
    {
        return makeFailure( FourierBsdfTableLoadStatus::FILE_NOT_FOUND,
                            "Unable to open Fourier BSDF table file \"" + fileName + "\"" );
    }

    const std::streamoff fileSize{ input.tellg() };
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

    FourierBsdfTable table{};
    int              unused{};
    if( !readInt32( input, table.flags ) || !readInt32( input, table.nMu ) || !readInt32( input, table.nCoefficients )
        || !readInt32( input, table.maxOrder ) || !readInt32( input, table.nChannels )
        || !readInt32( input, table.nBases ) || !readInt32( input, unused ) || !readInt32( input, unused )
        || !readInt32( input, unused ) || !readFloat( input, table.eta ) || !readInt32( input, unused )
        || !readInt32( input, unused ) || !readInt32( input, unused ) || !readInt32( input, unused ) )
    {
        return makeFailure( FourierBsdfTableLoadStatus::TRUNCATED, sectionDiagnostic( fileName, "metadata" ) );
    }

    if( table.flags != 1 || ( table.nChannels != 1 && table.nChannels != 3 ) || table.nBases != 1 )
    {
        return makeFailure( FourierBsdfTableLoadStatus::UNSUPPORTED, unsupportedDiagnostic( fileName, table ) );
    }
    if( table.nMu <= 0 || table.nCoefficients < 0 || table.maxOrder <= 0 )
    {
        return makeFailure( FourierBsdfTableLoadStatus::MALFORMED,
                            "Malformed Fourier BSDF table \"" + fileName + "\": invalid dimensions" );
    }

    const std::size_t nMu{ static_cast<std::size_t>( table.nMu ) };
    std::size_t       gridSize{};
    if( !checkedMultiply( nMu, nMu, gridSize ) )
    {
        return makeFailure( FourierBsdfTableLoadStatus::MALFORMED,
                            "Malformed Fourier BSDF table \"" + fileName + "\": dimension overflow" );
    }

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
    std::vector<int> offsetAndLength;
    if( !readInt32Vector( input, offsetAndLength, offsetPairCount ) )
    {
        return makeFailure( FourierBsdfTableLoadStatus::TRUNCATED,
                            sectionDiagnostic( fileName, "coefficient offsets" ) );
    }
    if( !readFloatVector( input, table.coefficients, static_cast<std::size_t>( table.nCoefficients ) ) )
    {
        return makeFailure( FourierBsdfTableLoadStatus::TRUNCATED, sectionDiagnostic( fileName, "coefficients" ) );
    }

    table.coefficientOffsets.resize( gridSize );
    table.coefficientCounts.resize( gridSize );
    table.zeroOrderCoefficients.resize( gridSize );
    for( std::size_t i = 0; i < gridSize; ++i )
    {
        const int offset{ offsetAndLength[2U * i] };
        const int length{ offsetAndLength[2U * i + 1U] };
        if( offset < 0 || length < 0 || length > table.maxOrder )
        {
            return makeFailure( FourierBsdfTableLoadStatus::MALFORMED,
                                "Malformed Fourier BSDF table \"" + fileName + "\": invalid coefficient span" );
        }
        std::size_t channelCoefficientCount{};
        if( !checkedMultiply( static_cast<std::size_t>( length ), static_cast<std::size_t>( table.nChannels ), channelCoefficientCount ) )
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

    const std::streamoff tableEnd{ input.tellg() };
    if( tableEnd >= 0 && fileSize >= tableEnd )
    {
        table.trailingByteCount = static_cast<std::size_t>( fileSize - tableEnd );
    }

    FourierBsdfTableLoadResult result{};
    result.status = FourierBsdfTableLoadStatus::SUCCESS;
    result.table  = std::move( table );
    return result;
}

}  // namespace demandPbrtScene
