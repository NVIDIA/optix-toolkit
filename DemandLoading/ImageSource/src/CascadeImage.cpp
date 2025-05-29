// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/CascadeImage.h>

#include <algorithm>
#include <vector>

namespace imageSource {

CascadeImage::CascadeImage( std::shared_ptr<ImageSource> backingImage, unsigned int minDim )
    : m_backingImage( backingImage )
    , m_minDim( minDim )
{
    m_backingMipLevel = 0;
    m_info = {};
    m_isOpen = false;
}

void CascadeImage::open( TextureInfo* info )
{
    if( !isOpen() )
    {
        TextureInfo backingInfo;
        m_backingImage->open( &backingInfo );

        unsigned int width  = backingInfo.width;
        unsigned int height = backingInfo.height;
        m_backingMipLevel   = 0;
        while( backingInfo.numMipLevels >= m_backingMipLevel && ( width/2 >= m_minDim && height/2 >= m_minDim ) )
        {
            m_backingMipLevel++;
            width /= 2;
            height /= 2;
        }

        m_info.width        = width;
        m_info.height       = height;
        m_info.format       = backingInfo.format;
        m_info.numChannels  = backingInfo.numChannels;
        m_info.numMipLevels = backingInfo.numMipLevels - m_backingMipLevel;
        m_info.isValid      = true;
        m_info.isTiled      = true;

        m_isOpen = true;
    }

    if( info != nullptr )
        *info = m_info;
}

bool CascadeImage::readMipTail( char*        dest,
                                unsigned int mipTailFirstLevel,
                                unsigned int /*numMipLevels*/,
                                const uint2* /*mipLevelDims*/,
                                CUstream     stream )
{
    if( !m_backingImage ) 
          return false;

    // Get mip level dimensions for the backing image
    unsigned int backingMipTailFirstLevel = mipTailFirstLevel + m_backingMipLevel;
    const TextureInfo& backingInfo = m_backingImage->getInfo();
    std::vector<uint2> backingMipLevelDims( backingInfo.numMipLevels );

    for( unsigned int mip = 0; mip < backingMipLevelDims.size(); ++mip )
        backingMipLevelDims[mip] = uint2{ std::max( backingInfo.width >> mip, 1U ), std::max( backingInfo.height >> mip, 1U ) };

    return m_backingImage->readMipTail( dest, backingMipTailFirstLevel, backingInfo.numMipLevels, &backingMipLevelDims[0], stream );
}

}  // namespace imageSource
