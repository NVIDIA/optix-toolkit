// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace demandPbrtScene {
namespace testing {

class FourierBsdfTableWriter
{
  public:
    explicit FourierBsdfTableWriter( const std::filesystem::path& fileName )
        : m_output{ fileName, std::ios::binary }
    {
        constexpr char header[8] = { 'S', 'C', 'A', 'T', 'F', 'U', 'N', '\x01' };
        m_output.write( header, sizeof( header ) );
    }

    void writeInt32( int value ) { writeUint32( static_cast<std::uint32_t>( value ) ); }

    void writeFloat( float value )
    {
        std::uint32_t bits{};
        std::memcpy( &bits, &value, sizeof( bits ) );
        writeUint32( bits );
    }

    void writeMetadata( int flags, int nMu, int nCoefficients, int maxOrder, int nChannels, int nBases )
    {
        writeInt32( flags );
        writeInt32( nMu );
        writeInt32( nCoefficients );
        writeInt32( maxOrder );
        writeInt32( nChannels );
        writeInt32( nBases );
        writeInt32( 0 );
        writeInt32( 0 );
        writeInt32( 0 );
        writeFloat( 1.0f );
        writeInt32( 0 );
        writeInt32( 0 );
        writeInt32( 0 );
        writeInt32( 0 );
    }

    static void writeMinimalTable( const std::filesystem::path& fileName,
                                   const std::vector<float>&    coefficients,
                                   int                          nChannels,
                                   int                          coefficientOffset,
                                   int                          coefficientCount )
    {
        FourierBsdfTableWriter output{ fileName };
        output.writeMetadata( 1, 1, static_cast<int>( coefficients.size() ), 1, nChannels, 1 );
        output.writeFloat( 1.0f );
        output.writeFloat( 1.0f );
        output.writeInt32( coefficientOffset );
        output.writeInt32( coefficientCount );
        for( float coefficient : coefficients )
        {
            output.writeFloat( coefficient );
        }
    }

    static void writeOrderShapeTable( const std::filesystem::path& fileName, int maxOrder )
    {
        constexpr int nMu{ 2 };
        constexpr int nChannels{ 3 };
        constexpr int gridSize{ nMu * nMu };
        const int     nCoefficients{ gridSize * nChannels * maxOrder };

        FourierBsdfTableWriter output{ fileName };
        output.writeMetadata( 1, nMu, nCoefficients, maxOrder, nChannels, 1 );
        output.writeFloat( -1.0f );
        output.writeFloat( 1.0f );
        output.writeFloat( 0.0f );
        output.writeFloat( 1.0f );
        output.writeFloat( 0.0f );
        output.writeFloat( 1.0f );
        for( int i = 0; i < gridSize; ++i )
        {
            output.writeInt32( i * nChannels * maxOrder );
            output.writeInt32( maxOrder );
        }
        for( int i = 0; i < nCoefficients; ++i )
        {
            output.writeFloat( i % maxOrder == 0 ? 1.0f : 0.0f );
        }
    }

  private:
    void writeUint32( std::uint32_t value )
    {
        const unsigned char bytes[] = {
            static_cast<unsigned char>( value & 0xffU ),
            static_cast<unsigned char>( ( value >> 8 ) & 0xffU ),
            static_cast<unsigned char>( ( value >> 16 ) & 0xffU ),
            static_cast<unsigned char>( ( value >> 24 ) & 0xffU ),
        };
        m_output.write( reinterpret_cast<const char*>( bytes ), sizeof( bytes ) );
    }

    std::ofstream m_output;
};

}  // namespace testing
}  // namespace demandPbrtScene
