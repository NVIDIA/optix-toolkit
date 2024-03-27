//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/Util/ImageBuffer.h>

#include <stb_image_write.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace otk {

static void savePPM( const unsigned char* Pix, const char* fname, int wid, int hgt, int chan )
{
    if( Pix == NULL || wid < 1 || hgt < 1 )
        throw std::runtime_error( "savePPM: Image is ill-formed. Not saving" );

    if( chan != 1 && chan != 3 && chan != 4 )
        throw std::runtime_error( "savePPM: Attempting to save image with channel count != 1, 3, or 4." );

    std::ofstream OutFile( fname, std::ios::out | std::ios::binary );
    if( !OutFile.is_open() )
        throw std::runtime_error( "savePPM: Could not open file for" );

    bool is_float = false;
    OutFile << 'P';
    OutFile << ( ( chan == 1 ? ( is_float ? 'Z' : '5' ) : ( chan == 3 ? ( is_float ? '7' : '6' ) : '8' ) ) )
            << std::endl;
    OutFile << wid << " " << hgt << std::endl << 255 << std::endl;

    OutFile.write( reinterpret_cast<char*>( const_cast<unsigned char*>( Pix ) ), wid*hgt*chan*( is_float ? 4 : 1 ) );
    OutFile.close();
}

size_t pixelFormatSize( BufferImageFormat format )
{
    switch( format )
    {
        case BufferImageFormat::UNSIGNED_BYTE4:
            return sizeof( char ) * 4;
        case BufferImageFormat::FLOAT3:
            return sizeof( float ) * 3;
        case BufferImageFormat::FLOAT4:
            return sizeof( float ) * 4;
        default:
            throw std::runtime_error( "otk::pixelFormatSize: Unrecognized buffer format" );
    }
}

static float toSRGB( float c )
{
    float invGamma = 1.0f / 2.4f;
    float powed    = std::pow( c, invGamma );
    return c < 0.0031308f ? 12.92f * c : 1.055f * powed - 0.055f;
}

static std::vector<std::uint8_t> getPixels( const ImageBuffer& image, bool disable_srgb_conversion )
{
    //
    // Note -- we are flipping image vertically as we write it into output buffer
    //
    const int32_t              width  = image.width;
    const int32_t              height = image.height;
    std::vector<unsigned char> pix( width * height * 3 );

    switch( image.pixel_format )
    {
        case BufferImageFormat::UNSIGNED_BYTE4:
        {
            for( int j = height - 1; j >= 0; --j )
            {
                for( int i = 0; i < width; ++i )
                {
                    const int32_t dst_idx = 3 * width * ( height - j - 1 ) + 3 * i;
                    const int32_t src_idx = 4 * width * j + 4 * i;
                    pix[dst_idx + 0]      = reinterpret_cast<uint8_t*>( image.data )[src_idx + 0];
                    pix[dst_idx + 1]      = reinterpret_cast<uint8_t*>( image.data )[src_idx + 1];
                    pix[dst_idx + 2]      = reinterpret_cast<uint8_t*>( image.data )[src_idx + 2];
                }
            }
        }
        break;

        case BufferImageFormat::FLOAT3:
        {
            for( int j = height - 1; j >= 0; --j )
            {
                for( int i = 0; i < width; ++i )
                {
                    const int32_t dst_idx = 3 * width * ( height - j - 1 ) + 3 * i;
                    const int32_t src_idx = 3 * width * j + 3 * i;
                    for( int elem = 0; elem < 3; ++elem )
                    {
                        const float   f = reinterpret_cast<float*>( image.data )[src_idx + elem];
                        const int32_t v = static_cast<int32_t>( 256.0f * ( disable_srgb_conversion ? f : toSRGB( f ) ) );
                        const int32_t c     = v < 0 ? 0 : v > 0xff ? 0xff : v;
                        pix[dst_idx + elem] = static_cast<uint8_t>( c );
                    }
                }
            }
        }
        break;

        case BufferImageFormat::FLOAT4:
        {
            for( int j = height - 1; j >= 0; --j )
            {
                for( int i = 0; i < width; ++i )
                {
                    const int32_t dst_idx = 3 * width * ( height - j - 1 ) + 3 * i;
                    const int32_t src_idx = 4 * width * j + 4 * i;
                    for( int elem = 0; elem < 3; ++elem )
                    {
                        const float   f = reinterpret_cast<float*>( image.data )[src_idx + elem];
                        const int32_t v = static_cast<int32_t>( 256.0f * ( disable_srgb_conversion ? f : toSRGB( f ) ) );
                        const int32_t c     = v < 0 ? 0 : v > 0xff ? 0xff : v;
                        pix[dst_idx + elem] = static_cast<uint8_t>( c );
                    }
                }
            }
        }
        break;

        default:
        {
            throw std::runtime_error( "otk::saveImage(): Unrecognized image buffer pixel format.\n" );
        }
    }

    return pix;
}

void saveImage( const char* fname, const ImageBuffer& image, bool disableSRGBConversion )
{
    const std::string filename( fname );
    const size_t dot = filename.find_last_of( '.' );
    if( dot == std::string::npos )
        throw std::runtime_error( "otk::saveImage(): Failed to determine filename extension" );

    const std::string ext = filename.substr( dot + 1 );
    if( ext == "PPM" || ext == "ppm" )
    {
        const std::vector<std::uint8_t> pix( getPixels( image, disableSRGBConversion ) );
        savePPM( pix.data(), filename.c_str(), image.width, image.height, 3 );
    }
    else if( ext == "png" || ext == "PNG" )
    {
        const std::vector<std::uint8_t> pix( getPixels( image, disableSRGBConversion ) );
        if( stbi_write_png( fname, image.width, image.height, 3, pix.data(), 3 * image.width ) == 0 )
        {
            throw std::runtime_error( "otk::saveImage(): Failed to write PNG image " + filename);
        }
    }
    else
    {
        throw std::runtime_error( "otk::saveImage(): Failed unsupported filetype '" + ext + "'");
    }
}

} // namespace otk
