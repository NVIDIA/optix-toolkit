// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <fstream>

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace imageSource {

const uint32_t DDS_MAGIC_NUMBER = 0x20534444U; // "DDS "
const uint32_t BC_BLOCK_WIDTH = 4;
const uint32_t BC_BLOCK_HEIGHT = 4;

enum DXGIFormats
{
    DXGI_FORMAT_BC6H_TYPELESS = 94,
    DXGI_FORMAT_BC6H_UF16 = 95,
    DXGI_FORMAT_BC6H_SF16 = 96,
    DXGI_FORMAT_BC7_TYPELESS = 97,
    DXGI_FORMAT_BC7_UNORM = 98,
    DXGI_FORMAT_BC7_UNORM_SRGB = 99
};

struct DDSFormatTranslation
{
    int dgxiFormat;
    const char* fourCCcode;
    CUarray_format format;
    int numChannels;
    int blockSize;
};

// FIXME: DO I need alternate codes? Also, should I handle non-bc compressed files?
const DDSFormatTranslation DDS_FORMAT_TRANSLATIONS[] = 
{
    {0, "DXT1", CU_AD_FORMAT_BC1_UNORM, 4, 8}, // FIXME: SRGB versions don't seem to work.
    {0, "DXT2", CU_AD_FORMAT_BC2_UNORM, 4, 16}, 
    {0, "DXT3", CU_AD_FORMAT_BC2_UNORM, 4, 16},
    {0, "DXT4", CU_AD_FORMAT_BC3_UNORM, 4, 16},
    {0, "DXT5", CU_AD_FORMAT_BC3_UNORM, 4, 16},
    {0, "ATI1", CU_AD_FORMAT_BC4_UNORM, 1, 8},
    {0, "ATI2", CU_AD_FORMAT_BC5_UNORM, 2, 16},
    {DXGI_FORMAT_BC6H_TYPELESS, "DX10", CU_AD_FORMAT_BC6H_UF16, 3, 16},
    {DXGI_FORMAT_BC6H_UF16, "DX10", CU_AD_FORMAT_BC6H_UF16, 3, 16},
    {DXGI_FORMAT_BC6H_SF16, "DX10", CU_AD_FORMAT_BC6H_SF16, 3, 16},
    {DXGI_FORMAT_BC7_TYPELESS, "DX10", CU_AD_FORMAT_BC7_UNORM, 4, 16},
    {DXGI_FORMAT_BC7_UNORM, "DX10", CU_AD_FORMAT_BC7_UNORM, 4, 16},
    {DXGI_FORMAT_BC7_UNORM_SRGB, "DX10", CU_AD_FORMAT_BC7_UNORM_SRGB, 4, 16},
};

struct DDSPixelFormat 
{
    uint32_t sizeCheck; // always 32
    uint32_t flags;
    char fourCCcode[4];
    uint32_t rgbBitCount;
    uint32_t rBitMask;
    uint32_t gBitMask;
    uint32_t bBitMask;
    uint32_t aBitMask;
};

struct DDSFileHeader
{
    uint32_t magicNumber; // "DDS " (0x20534444)
    uint32_t sizeCheck; // always 124
    uint32_t flags;
    uint32_t height;
    uint32_t width;
    uint32_t pitchOrLinearSize;
    uint32_t depth;
    uint32_t mipMapCount;
    uint32_t reserved1[11];
    DDSPixelFormat pixelFormat;
    uint32_t caps1;
    uint32_t caps2;
    uint32_t caps3;
    uint32_t caps4;
    uint32_t reserved2;
};

struct DDSHeaderExtension
{
    uint32_t dxgiFormat;
    uint32_t resourceDimension;
    uint32_t miscFlag;
    uint32_t arraySize;
    uint32_t miscFlags2;
};

/// DDSImageReader, reads direct draw surface (.dds) image files that 
/// encode BC1-BC7 compressed formats.
class DDSImageReader : public ImageSourceBase
{
  public:
    /// Create a test image with the specified dimensions.
    DDSImageReader( const std::string& fileName, bool readBaseColor );

    /// The destructor is virtual.
    ~DDSImageReader() override { close(); }

    /// The open method simply initializes the given image info struct.
    void open( imageSource::TextureInfo* info ) override;

    /// Close the image
    void close() override { m_file.close(); }

    /// Check if image is currently open.
    bool isOpen() const override { return m_file.is_open(); }

    /// Get the image info.  Valid only after calling open().
    const imageSource::TextureInfo& getInfo() const override { return m_info; }

    /// Return the mode in which the image fills part of itself
    CUmemorytype getFillType() const override { return CU_MEMORYTYPE_HOST; }

    /// Read the specified tile or mip level, returning the data in dest. 
    bool readTile( char* dest, unsigned int mipLevel, const imageSource::Tile& tile, CUstream stream ) override;

    /// Read the specified mipLevel.  Returns true for success.
    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream ) override;

    /// Read the base color of the image (1x1 mip level) as a float4. Returns true on success.
    bool readBaseColor( float4& dest ) override;

    int getMipLevelFileOffsetInBytes( int mipLevel );
    int getMipLevelSizeInBytes( int mipLevel );
    int getMipLevelWidthInBytes( int mipLevel );

  private:
    std::mutex m_mutex;
    std::mutex m_mipCacheMutex;
    std::string m_fileName;
    std::ifstream m_file;
    int m_fileHeaderOffset;
    int m_blockSizeInBytes;

    bool m_readBaseColor;
    float4 m_baseColor;
    imageSource::TextureInfo m_info;
    std::vector<std::vector<char>> m_mipCache;
};

}  // namespace imageSource
