//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <gmock/gmock.h>

#include <memory>
#include <ostream>

#define OUTPUT_ENUM( enum_ )                                                                                           \
    case enum_:                                                                                                        \
        return str << #enum_ << " (" << static_cast<int>( enum_ ) << ')'

inline std::ostream& operator<<( std::ostream& str, CUarray_format val )
{
    switch( val )
    {
        OUTPUT_ENUM( CU_AD_FORMAT_UNSIGNED_INT8 );
        OUTPUT_ENUM( CU_AD_FORMAT_UNSIGNED_INT16 );
        OUTPUT_ENUM( CU_AD_FORMAT_UNSIGNED_INT32 );
        OUTPUT_ENUM( CU_AD_FORMAT_SIGNED_INT8 );
        OUTPUT_ENUM( CU_AD_FORMAT_SIGNED_INT16 );
        OUTPUT_ENUM( CU_AD_FORMAT_SIGNED_INT32 );
        OUTPUT_ENUM( CU_AD_FORMAT_HALF );
        OUTPUT_ENUM( CU_AD_FORMAT_FLOAT );
        OUTPUT_ENUM( CU_AD_FORMAT_NV12 );
        OUTPUT_ENUM( CU_AD_FORMAT_UNORM_INT8X1 );
        OUTPUT_ENUM( CU_AD_FORMAT_UNORM_INT8X2 );
        OUTPUT_ENUM( CU_AD_FORMAT_UNORM_INT8X4 );
        OUTPUT_ENUM( CU_AD_FORMAT_UNORM_INT16X1 );
        OUTPUT_ENUM( CU_AD_FORMAT_UNORM_INT16X2 );
        OUTPUT_ENUM( CU_AD_FORMAT_UNORM_INT16X4 );
        OUTPUT_ENUM( CU_AD_FORMAT_SNORM_INT8X1 );
        OUTPUT_ENUM( CU_AD_FORMAT_SNORM_INT8X2 );
        OUTPUT_ENUM( CU_AD_FORMAT_SNORM_INT8X4 );
        OUTPUT_ENUM( CU_AD_FORMAT_SNORM_INT16X1 );
        OUTPUT_ENUM( CU_AD_FORMAT_SNORM_INT16X2 );
        OUTPUT_ENUM( CU_AD_FORMAT_SNORM_INT16X4 );
        OUTPUT_ENUM( CU_AD_FORMAT_BC1_UNORM );
        OUTPUT_ENUM( CU_AD_FORMAT_BC1_UNORM_SRGB );
        OUTPUT_ENUM( CU_AD_FORMAT_BC2_UNORM );
        OUTPUT_ENUM( CU_AD_FORMAT_BC2_UNORM_SRGB );
        OUTPUT_ENUM( CU_AD_FORMAT_BC3_UNORM );
        OUTPUT_ENUM( CU_AD_FORMAT_BC3_UNORM_SRGB );
        OUTPUT_ENUM( CU_AD_FORMAT_BC4_UNORM );
        OUTPUT_ENUM( CU_AD_FORMAT_BC4_SNORM );
        OUTPUT_ENUM( CU_AD_FORMAT_BC5_UNORM );
        OUTPUT_ENUM( CU_AD_FORMAT_BC5_SNORM );
        OUTPUT_ENUM( CU_AD_FORMAT_BC6H_UF16 );
        OUTPUT_ENUM( CU_AD_FORMAT_BC6H_SF16 );
        OUTPUT_ENUM( CU_AD_FORMAT_BC7_UNORM );
        OUTPUT_ENUM( CU_AD_FORMAT_BC7_UNORM_SRGB );
    }
    return str << "? (" << static_cast<int>( val ) << ')';
}

inline std::ostream& operator<<( std::ostream& str, CUmemorytype val )
{
    switch( val )
    {
        OUTPUT_ENUM( CU_MEMORYTYPE_HOST );
        OUTPUT_ENUM( CU_MEMORYTYPE_DEVICE );
        OUTPUT_ENUM( CU_MEMORYTYPE_ARRAY );
        OUTPUT_ENUM( CU_MEMORYTYPE_UNIFIED );
    }
    return str << "? (" << static_cast<int>( val ) << ')';
}

#undef OUTPUT_ENUM

namespace imageSource {

inline bool operator==( const Tile& lhs, const Tile& rhs )
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.width == rhs.width && lhs.height == rhs.height;
}

inline bool operator!=( const Tile& lhs, const Tile& rhs )
{
    return !( lhs == rhs );
}

inline std::ostream& operator<<( std::ostream& str, const Tile& val )
{
    return str << "Tile{" << val.x << ',' << val.y << ' ' << val.width << 'x' << val.height << '}';
}

inline void PrintTo( const Tile& val, std::ostream* str )
{
    *str << val;
}

inline std::ostream& operator<<( std::ostream& str, const TextureInfo& val )
{
    return str << "TextureInfo{ " << val.width << ", " << val.height << ", " << val.format << ", " << val.numChannels << ", "
               << val.numMipLevels << ", " << std::boolalpha << val.isValid << ", " << std::boolalpha << val.isTiled << " }";
}

inline void PrintTo( const TextureInfo& val, std::ostream* str )
{
    *str << val;
}

}  // namespace imageSource

namespace otk {
namespace testing {

class MockImageSource : public ::testing::StrictMock<imageSource::ImageSource>
{
  public:
    ~MockImageSource() override = default;

    MOCK_METHOD( void, open, (imageSource::TextureInfo*), ( override ) );
    MOCK_METHOD( void, close, (), ( override ) );
    MOCK_METHOD( bool, isOpen, (), ( const, override ) );
    MOCK_METHOD( const imageSource::TextureInfo&, getInfo, (), ( const, override ) );
    MOCK_METHOD( CUmemorytype, getFillType, (), ( const, override ) );
    MOCK_METHOD( bool, readTile, ( char*, unsigned, const imageSource::Tile&, CUstream ), ( override ) );
    MOCK_METHOD( bool, readMipLevel, ( char*, unsigned, unsigned, unsigned, CUstream ), ( override ) );
    MOCK_METHOD( bool, readMipTail, ( char*, unsigned, unsigned, const uint2*, unsigned, CUstream ), ( override ) );
    MOCK_METHOD( bool, readBaseColor, (float4&), ( override ) );
    MOCK_METHOD( unsigned int, getTileWidth, (), ( const, override ) );
    MOCK_METHOD( unsigned int, getTileHeight, (), ( const, override ) );
    MOCK_METHOD( unsigned long long, getNumTilesRead, (), ( const, override ) );
    MOCK_METHOD( unsigned long long, getNumBytesRead, (), ( const, override ) );
    MOCK_METHOD( double, getTotalReadTime, (), ( const, override ) );
    MOCK_METHOD( bool, hasCascade, (), ( const override ) );
};

using MockImageSourcePtr = std::shared_ptr<MockImageSource>;

}  // namespace testing
}  // namespace otk
