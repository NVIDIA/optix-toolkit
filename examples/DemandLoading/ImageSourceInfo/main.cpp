// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/TextureInfo.h>
#include <OptiXToolkit/ImageSources/ImageSources.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#define ENUM_CASE( id_ )                                                                                               \
    case CU_AD_FORMAT_##id_:                                                                                           \
        return str << #id_

std::ostream& operator<<( std::ostream& str, CUarray_format value )
{
    switch( value )
    {
        ENUM_CASE( UNSIGNED_INT8 );
        ENUM_CASE( UNSIGNED_INT16 );
        ENUM_CASE( UNSIGNED_INT32 );
        ENUM_CASE( SIGNED_INT8 );
        ENUM_CASE( SIGNED_INT16 );
        ENUM_CASE( SIGNED_INT32 );
        ENUM_CASE( HALF );
        ENUM_CASE( FLOAT );
        ENUM_CASE( NV12 );
        ENUM_CASE( UNORM_INT8X1 );
        ENUM_CASE( UNORM_INT8X2 );
        ENUM_CASE( UNORM_INT8X4 );
        ENUM_CASE( UNORM_INT16X1 );
        ENUM_CASE( UNORM_INT16X2 );
        ENUM_CASE( UNORM_INT16X4 );
        ENUM_CASE( SNORM_INT8X1 );
        ENUM_CASE( SNORM_INT8X2 );
        ENUM_CASE( SNORM_INT8X4 );
        ENUM_CASE( SNORM_INT16X1 );
        ENUM_CASE( SNORM_INT16X2 );
        ENUM_CASE( SNORM_INT16X4 );
        ENUM_CASE( BC1_UNORM );
        ENUM_CASE( BC1_UNORM_SRGB );
        ENUM_CASE( BC2_UNORM );
        ENUM_CASE( BC2_UNORM_SRGB );
        ENUM_CASE( BC3_UNORM );
        ENUM_CASE( BC3_UNORM_SRGB );
        ENUM_CASE( BC4_UNORM );
        ENUM_CASE( BC4_SNORM );
        ENUM_CASE( BC5_UNORM );
        ENUM_CASE( BC5_SNORM );
        ENUM_CASE( BC6H_UF16 );
        ENUM_CASE( BC6H_SF16 );
        ENUM_CASE( BC7_UNORM );
        ENUM_CASE( BC7_UNORM_SRGB );
    }
    return str << "?unknown format " << std::to_string( value );
}
#undef ENUM_CASE

namespace {

class ImageSourceInfo
{
  public:
    ImageSourceInfo( const char* texture )
        : m_texture( texture )
    {
    }
    void printInfo();

  private:
    std::string m_texture;
};

void ImageSourceInfo::printInfo()
{
    std::shared_ptr<imageSource::ImageSource> image{ imageSources::createImageSource( m_texture ) };
    imageSource::TextureInfo                  info{};
    image->open( &info );

    // clang-format off
    std::cout << "TextureInfo: " << m_texture << "\n"
        "         width: " << info.width << "\n"
        "        height: " << info.height << "\n"
        "        format: " << info.format << "\n"
        "      channels: " << info.numChannels << "\n"
        "    mip levels: " << info.numMipLevels << "\n"
        "         valid: " << std::boolalpha << info.isValid << "\n"
        "         tiled: " << std::boolalpha << info.isTiled << "\n";
    // clang-format on
}

int usage( const char* program )
{
    std::cerr << "Usage: " << program << " <texture>\n";
    return -1;
}

}  // namespace

int main( int argc, char* argv[] )
{
    try
    {
        if( argc != 2 )
        {
            return usage( argv[0] );
        }
        ImageSourceInfo( argv[1] ).printInfo();
    }
    catch( const std::exception& bang )
    {
        std::cerr << bang.what() << '\n';
        return 1;
    }
    catch( ... )
    {
        std::cerr << "Unknown exception\n";
        return 2;
    }
    return 0;
}
