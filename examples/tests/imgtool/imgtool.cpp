// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "pngcompare.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <utility>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef unsigned char u8;

struct OptPpmCompare
{
  std::string  file1;
  std::string  file2;
  std::string  diff_file;
  float        diff_threshold = 0;
  float        allowed_percentage = 0;
};

struct OptTextCompare
{
    std::string file1;
    std::string file2;
};

struct OptPpm2Png
{
  std::string  ppmfile;
  std::string  pngfile;
  std::string  default_pngfile;
};

struct Options
{
  std::string     mode;
  OptPpmCompare   ppmcompare;
  OptPpm2Png      ppm2png;
  OptPngCompare   pngcompare;
  OptTextCompare  textcompare;
} opts;

struct ImageData
{
  int              resx = 0;
  int              resy = 0;
  std::vector<u8>  data;
};

void usage( const char* appname )
{
  printf( "\nusage: %s <mode> [options]\n\n", appname );
  printf( "  mode \"ppmcompare\":\n" );
  printf( "    %s ppmcompare <file1> <file2> <diff_file> <diff_threshold> <allowed_percentage>\n", appname );
  printf( "  mode \"ppm2png\":\n" );
  printf( "    %s ppm2png <ppmfile> <pngfile> [default_pngfile]\n", appname );
  printf( "  mode \"pngcompare\":\n" );
  printf( "    %s pngcompare <file1> <file2> <diff_file> <diff_threshold> <allowed_percentage>\n", appname );
  printf( "  mode \"textcompare\":\n" );
  printf( "    %s textcompare <file1> <file2>\n", appname );
  printf( "\n" );
  exit( -1 );
}

void parse_options( int argc, char** argv )
{
  if( argc < 2 )
    usage( argv[0] );

  opts.mode = argv[1];

  if( opts.mode == "ppmcompare" )
  {
    if( argc != 7 )
      usage( argv[0] );

    opts.ppmcompare.file1 = argv[2];
    opts.ppmcompare.file2 = argv[3];
    opts.ppmcompare.diff_file = argv[4];
    opts.ppmcompare.diff_threshold = static_cast<float>( atof( argv[5] ) );
    opts.ppmcompare.allowed_percentage = static_cast<float>( atof( argv[6] ) );
  }
  else if( opts.mode == "pngcompare" )
  {
    if( argc != 7 )
      usage( argv[0] );

    opts.pngcompare.file1 = argv[2];
    opts.pngcompare.file2 = argv[3];
    opts.pngcompare.diffFile = argv[4];
    opts.pngcompare.diffThreshold = static_cast<float>( atof( argv[5] ) );
    opts.pngcompare.allowedPercentage = static_cast<float>( atof( argv[6] ) );
  }
  else if( opts.mode == "ppm2png" )
  {
    if( argc < 4 || argc > 5 )
      usage( argv[0] );

    opts.ppm2png.ppmfile = argv[2];
    opts.ppm2png.pngfile = argv[3];
    if( argc > 4 ) opts.ppm2png.default_pngfile = argv[4];
  }
  else if( opts.mode == "textcompare" )
  {
      if( argc != 4 )
          usage( argv[0] );

      opts.textcompare.file1              = argv[2];
      opts.textcompare.file2              = argv[3];
  }
  else
  {
    usage( argv[0] );
  }
}

void copy_file( const char* from, const char* to )
{
  std::ifstream src( from, std::ios::binary );
  std::ofstream dst( to,   std::ios::binary );
  dst << src.rdbuf();
}

bool readppm( std::ifstream& ifs, ImageData& img )
{
  std::string s;

  // read magic string
  if( !ifs.good() )
    return false;
  ifs >> s;
  if( s != "P6" )
    return false;

  // read x resolution
  if( !ifs.good() )
    return false;
  ifs >> s;
  img.resx = atoi( s.c_str() );

  // read y resolution
  if( !ifs.good() )
    return false;
  ifs >> s;
  img.resy = atoi( s.c_str() );

  // read max value
  if( !ifs.good() )
    return false;
  ifs >> s;

  // read single whitespace character
  if( !ifs.good() )
    return false;
  ifs.get();

  // read data
  if( !ifs.good() )
    return false;
  const int nbytes = img.resx * img.resy * 3;
  img.data.resize( nbytes );
  ifs.read( (char*)img.data.data(), nbytes );

  return !ifs.fail();
}

bool writeppm( std::ofstream& ofs, const ImageData& img )
{
  if( !ofs.good() )
    return false;
  ofs << "P6\n";

  if( !ofs.good() )
    return false;
  ofs << img.resx << " " << img.resy << "\n";

  if( !ofs.good() )
    return false;
  ofs << "255\n";

  if( !ofs.good() )
    return false;
  ofs.write( (const char*)img.data.data(), img.data.size() );

  return !ofs.fail();
}

int ppmcompare()
{
  const std::string file1 = opts.ppmcompare.file1;
  const std::string file2 = opts.ppmcompare.file2;

  printf( "Comparing images \"%s\" and \"%s\" with a per-channel difference threshold of %f and %f%% of pixels allowed to differ\n",
      file1.c_str(), file2.c_str(), opts.ppmcompare.diff_threshold, opts.ppmcompare.allowed_percentage );


  // Load files

  std::ifstream ifs1( file1.c_str(), std::ifstream::in | std::ifstream::binary );
  std::ifstream ifs2( file2.c_str(), std::ifstream::in | std::ifstream::binary );
  if( !ifs1.is_open() ) {
    printf( "Could not open file: %s\n", file1.c_str() );
    return 1;
  }
  if( !ifs2.is_open() ) {
    printf( "Could not open file: %s\n", file2.c_str() );
    return 1;
  }

  ImageData img1;
  ImageData img2;

  if( !readppm( ifs1, img1 ) ) {
    printf( "Error reading PPM file: %s\n", file1.c_str() );
    return 1;
  }

  if( !readppm( ifs2, img2 ) ) {
    printf( "Error reading PPM file: %s\n", file2.c_str() );
    return 1;
  }


  // Compare headers

  if( img1.resx != img2.resx || img1.resy != img2.resy || img1.data.size() != img2.data.size() ) {
    printf( "Images are of different size\n" );
    return 1;
  }

  
  // Compare content

  ImageData diffimg;
  diffimg.data.resize( img1.data.size() );
  diffimg.resx = img1.resx;
  diffimg.resy = img1.resy;

  int diffcnt = 0;

  for( int i=0; i<img1.resx * img1.resy; ++i )
  {
    float maxchanneldiff = 0;

    for( int j=0; j<3; ++j ) {
      const float diff = fabsf( float(img1.data[i*3+j]) - float(img2.data[i*3+j]) );
      maxchanneldiff = std::max( diff, maxchanneldiff );
    }

    if( maxchanneldiff > opts.ppmcompare.diff_threshold )
    {
      diffcnt++;
      diffimg.data[i*3+0] = 255;
      diffimg.data[i*3+1] = 0;
      diffimg.data[i*3+2] = 0;
    }
    else if( maxchanneldiff > 0 )
    {
      diffimg.data[i*3+0] = 255;
      diffimg.data[i*3+1] = 255;
      diffimg.data[i*3+2] = 0;
    }
  }

  const float diff_percentage = 100.0f * float(diffcnt) / float(img1.resx*img1.resy);
  printf( "%f%% of pixels exceed diff threshold\n", diff_percentage );


  // Write diff image

  const std::string diff_file = opts.ppmcompare.diff_file;
  std::ofstream ofs( diff_file.c_str(), std::ofstream::out | std::ofstream::binary );
  if( !ofs.is_open() ) {
    printf( "Could not open diff image: %s\n", diff_file.c_str() );
    return 1;
  }

  if( !writeppm( ofs, diffimg ) ) {
    printf( "Error writing diff image: %s\n", diff_file.c_str() );
    return 1;
  }


  // Finish

  if( diff_percentage > opts.ppmcompare.allowed_percentage ) {
    printf( "Images considered different\n" );
    return 1;
  }

  printf( "Images considered equivalent\n" );
  return 0;
}

int ppm2png()
{
  const std::string ppmfile = opts.ppm2png.ppmfile;
  const std::string pngfile = opts.ppm2png.pngfile;
  const std::string default_pngfile = opts.ppm2png.default_pngfile;
  const bool has_default = !default_pngfile.empty();

  printf( "Converting \"%s\" to \"%s\"\n", ppmfile.c_str(), pngfile.c_str() );


  // Load PPM file

  std::ifstream ifs( ppmfile.c_str(), std::ifstream::in | std::ifstream::binary );
  if( !ifs.is_open() ) {
    printf( "Could not open file: %s\n", ppmfile.c_str() );
    if( has_default )
      copy_file( default_pngfile.c_str(), pngfile.c_str() );
    return 1;
  }

  ImageData img;
  if( !readppm( ifs, img ) ) {
    printf( "Error reading PPM file: %s\n", ppmfile.c_str() );
    if( has_default )
      copy_file( default_pngfile.c_str(), pngfile.c_str() );
    return 1;
  }


  // Convert to PNG and save

  const int success = stbi_write_png( pngfile.c_str(), img.resx, img.resy, 3, img.data.data(), 3*img.resx );
  if( !success ) {
    printf( "Error writing PNG file: %s\n", pngfile.c_str() );
    if( has_default )
      copy_file( default_pngfile.c_str(), pngfile.c_str() );
    return 1;
  }
  
  return 0;
}

int textcompare()
{
    const std::string file1 = opts.textcompare.file1;
    const std::string file2 = opts.textcompare.file2;

    printf( "Comparing files \"%s\" and \"%s\"\n", file1.c_str(), file2.c_str() );

    // Load files
    std::ifstream ifs1( file1.c_str(), std::ifstream::in | std::ifstream::binary );
    std::ifstream ifs2( file2.c_str(), std::ifstream::in | std::ifstream::binary );
    if( !ifs1.is_open() )
    {
        printf( "Could not open file: %s\n", file1.c_str() );
        return 1;
    }
    if( !ifs2.is_open() )
    {
        printf( "Could not open file: %s\n", file2.c_str() );
        return 1;
    }

    // Compare sizes
    if( ifs1.tellg() != ifs2.tellg() )
    {
        printf( "File sizes not equal\n" );
        return 1;
    }

    // Compare content
    ifs1.seekg( 0, std::ifstream::beg );
    ifs2.seekg( 0, std::ifstream::beg );
    if( !std::equal( std::istreambuf_iterator<char>( ifs1.rdbuf() ), std::istreambuf_iterator<char>(),
                     std::istreambuf_iterator<char>( ifs2.rdbuf() ) ) )
    {
        printf( "Files considered different \n" );
        return 1;
    }

    // Finish
    printf( "Files considered equivalent\n" );
    return 0;
}

int main( int argc, char** argv )
{
  parse_options( argc, argv );
  
  if( opts.mode == "ppmcompare" )
    return ppmcompare();
  if( opts.mode == "ppm2png" )
    return ppm2png();
  if( opts.mode == "pngcompare" )
    return pngcompare( opts.pngcompare );
  if( opts.mode == "textcompare" )
      return textcompare();

  return -2;
}

