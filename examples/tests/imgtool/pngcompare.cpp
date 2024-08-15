// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "pngcompare.h"

#include <algorithm>
#include <string>
#include <vector>

#include <math.h>
#include <stdio.h>

#include "stb_image.h"
#include "stb_image_write.h"

class StbImageReader
{
public:
  StbImageReader( const std::string& file )
    : m_file( file )
  {
    m_data = stbi_load( m_file.c_str(), &m_width, &m_height, &m_numChannels, 3 );
    if( m_data == nullptr )
    {
      printf( "Loading image '%s' failed: %s\n", m_file.c_str(), stbi_failure_reason() );
    }
  }
  ~StbImageReader()
  {
    if( m_data != nullptr )
    {
      stbi_image_free( m_data );
    }
  }

  const std::string& file() const { return m_file; };
  int width() const { return m_width; }
  int height() const { return m_height; }
  int channels() const { return m_numChannels; }
  bool valid() const { return m_data != nullptr; }
  unsigned char getPixel( int x, int y, int channel ) const { return *( m_data + y * m_width * m_numChannels + x * m_numChannels + channel ); }
  unsigned char red( int x, int y ) const { return getPixel( x, y, 0 ); }
  unsigned char green( int x, int y ) const { return getPixel( x, y, 1 ); }
  unsigned char blue( int x, int y ) const { return getPixel( x, y, 2 ); }
  
private:
  std::string    m_file;
  int            m_width =0;
  int            m_height = 0;
  int            m_numChannels = 0;
  unsigned char* m_data = nullptr;
};

class StbImageWriter
{
public:
  StbImageWriter( const std::string& file, int width, int height, int numChannels )
    : m_file( file )
    , m_width( width )
    , m_height( height )
    , m_numChannels( numChannels )
    , m_data( m_width * m_height * m_numChannels )
  {
  }
  ~StbImageWriter() = default;

  const std::string& file() const { return m_file; }
  int width() const { return m_width; }
  int height() const { return m_height; }
  int channels() const { return m_numChannels; }
  void setPixel( int x, int y, int chan, unsigned char val )
  {
    m_data[y * m_width * m_numChannels + x * m_numChannels + chan] = val;
  }
  bool commit()
  {
    const int stride = m_width * m_numChannels;
    const int result = stbi_write_png( m_file.c_str(), m_width, m_height, m_numChannels, m_data.data(), stride );
    return result != 0;
  }

private:
  std::string    m_file;
  int            m_width =0;
  int            m_height = 0;
  int            m_numChannels = 0;
  std::vector<unsigned char> m_data;
};

int pngcompare( const OptPngCompare& opts )
{
  printf( "Comparing images \"%s\" and \"%s\" with a per-channel difference threshold of %f and %f%% of pixels allowed to differ\n",
    opts.file1.c_str(), opts.file2.c_str(), opts.diffThreshold, opts.allowedPercentage );

  StbImageReader image1( opts.file1 );
  StbImageReader image2( opts.file2 );
  if( !image1.valid() || !image2.valid() )
  {
    return 1;
  }
  if( image1.width() != image2.width() || image1.height() != image2.height() )
  {
    printf("Image dimensions don't match: %s (%d,%d) != %s (%d,%d)\n", image1.file().c_str(), image1.width(), image1.height(), image2.file().c_str(), image2.width(), image2.height() );
    return 1;
  }
  if( image1.channels() != image2.channels() )
  {
    printf("Image channel count doesn't match: %s %d != %s %d\n", image1.file().c_str(), image1.channels(), image2.file().c_str(), image2.channels() );
    return 1;
  }

  StbImageWriter diffImage( opts.diffFile, image1.width(), image1.height(), image1.channels() );
  int diffCount = 0;
  for( int y = 0; y < image1.height(); ++y )
  {
    for( int x = 0; x < image1.width(); ++x )
    {
      float maxChannelDiff = 0;
      for( int c = 0; c < 3; ++c )
      {
        const float diff = fabsf( static_cast<float>( image1.getPixel( x, y, c ) ) - static_cast<float>( image2.getPixel( x, y, c ) ) );
        maxChannelDiff = std::max( diff, maxChannelDiff );
      }

      if( maxChannelDiff > opts.diffThreshold )
      {
        diffCount++;
        diffImage.setPixel( x, y, 0, 255 );
        diffImage.setPixel( x, y, 1, 0 );
        diffImage.setPixel( x, y, 2, 0 );
      }
      else if( maxChannelDiff > 0 )
      {
        diffImage.setPixel( x, y, 0, 255 );
        diffImage.setPixel( x, y, 1, 255 );
        diffImage.setPixel( x, y, 2, 0 );
      }
    }
  }

  const float diffPercentage = 100.0f * static_cast<float>( diffCount ) / static_cast<float>( image1.width() * image1.height() );
  printf( "%f%% of pixels exceed diff threshold\n", diffPercentage );

  if( !diffImage.commit() )
  {
    printf( "Error writing diff image: %s\n", diffImage.file().c_str() );
    return 1;
  }

  if( diffPercentage > opts.allowedPercentage )
  {
    printf( "Images considered different\n" );
    return 1;
  }

  printf( "Images considered equivalent\n" );
  return 0;
}
