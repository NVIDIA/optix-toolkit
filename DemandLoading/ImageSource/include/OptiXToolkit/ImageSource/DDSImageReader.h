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
const uint32_t D3D11_RESOURCE_MISC_TILED = 0x40000L;
const uint32_t MISC_TILED_PIXEL_LAYOUT = 0x10000000;

const uint32_t TILE_SIZE_IN_BYTES = 65536;

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

    /// Read the mip tail
    bool readMipTail( char* dest, unsigned int mipTailFirstLevel, unsigned int numMipLevels, 
                      const uint2* mipLevelDims, CUstream stream ) override;

    /// Whether the file is tiled.  Valid only after calling open().
    bool isFileTiled() { return m_fileIsTiled; }

    /// Get the width of a tile that would be used for CUDA sparse textures.
    unsigned int getTileWidth() const override { return ( m_blockSizeInBytes == 8 ) ? 512u : 256u; }

    /// Get the height of a tile that would be used for CUDA sparse textures.
    unsigned int getTileHeight() const override { return 256u; }

    /// Get the mip level width in bytes (for a flat file)
    int getMipLevelWidthInBytes( int mipLevel );

    /// Get the size of a mip level in bytes
    int getMipLevelSizeInBytes( int mipLevel );

    /// Save as a tiled file which can be read from disk in a tile-wise fashion.
    bool saveAsTiledFile( const char* fileName );

    /// Returns the number of tiles that have been read.
    unsigned long long getNumTilesRead() const override { return m_numTilesRead; }

    /// Returns the number of bytes that have been read.
    unsigned long long getNumBytesRead() const override { return m_numBytesRead; }

    /// Returns the time in seconds spent reading image tiles.
    double getTotalReadTime() const override { return m_totalReadTime; }

  private:
    std::mutex m_mutex;
    std::mutex m_mipCacheMutex;
    std::string m_fileName;
    std::ifstream m_file;
    int m_fileHeaderOffset;
    int m_blockSizeInBytes;

    bool m_fileIsTiled;
    bool m_readBaseColor;
    float4 m_baseColor;
    imageSource::TextureInfo m_info;
    std::vector<std::vector<char>> m_mipCache;

    DDSFileHeader m_ddsFileHeader{};
    DDSHeaderExtension m_ddsHeaderExtension{};

    // Stats
    std::mutex         m_statsMutex;
    unsigned long long m_numTilesRead  = 0;
    unsigned long long m_numBytesRead  = 0;
    double             m_totalReadTime = 0.0;

    // Saving the file
    int getMipTailStartLevel();
    int getMipTailSize();

    // Reading flat (non-tiled) files
    bool readTileFlat( char* dest, unsigned int mipLevel, const Tile& tile );
    bool readMipLevelFlat( char* dest, unsigned int mipLevel );
    int getMipLevelOffsetInBytesFlat( int mipLevel );
    int getMipLevelSizeInBytesFlat( int mipLevel );

    // Reading tiled files
    bool readTileTiled( char* dest, unsigned int mipLevel, const Tile& tile );
    bool readMipLevelTiled( char* dest, unsigned int mipLevel );
    bool readMipTailTiled( char* dest, unsigned int mipTailFirstLevel );
    int getMipLevelOffsetInBytesTiled( int mipLevel );
    int getMipLevelWidthInTiles( int mipLevel ) { return ( ( m_info.width >> mipLevel ) + getTileWidth() - 1 ) / getTileWidth(); }
    int getMipLevelHeightInTiles( int mipLevel ) { return ( ( m_info.height >> mipLevel ) + getTileHeight() - 1 ) / getTileHeight(); }
    int getMipLevelSizeInBytesTiled( int mipLevel );
    int getTileOffsetInBytesTiled( int mipLevel, const Tile& tile );
};

}  // namespace imageSource
