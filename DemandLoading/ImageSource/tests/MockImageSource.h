// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
    MOCK_METHOD( bool, readMipTail, ( char*, unsigned, unsigned, const uint2*, CUstream ), ( override ) );
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
