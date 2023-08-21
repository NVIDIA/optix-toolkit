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

#include <vector>
#include <OptiXToolkit/ImageSource/CascadeImage.h>

namespace imageSource {

CascadeImage::CascadeImage( std::shared_ptr<imageSource::ImageSource> backingImage, unsigned int minDim )
    : m_backingImage( backingImage )
    , m_minDim( minDim )
{
    m_backingMipLevel = 0;
    m_info = {};
    m_isOpen = false;
}

void CascadeImage::open( imageSource::TextureInfo* info )
{
    if( !isOpen() )
    {
        imageSource::TextureInfo backingInfo;
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
                                unsigned int pixelSizeInBytes,
                                CUstream     stream )
{
    if( !m_backingImage ) 
          return false;

    // Get mip level dimensions for the backing image
    unsigned int backingMipTailFirstLevel = mipTailFirstLevel + m_backingMipLevel;
    const imageSource::TextureInfo& backingInfo = m_backingImage->getInfo();
    std::vector<uint2> backingMipLevelDims( backingInfo.numMipLevels );

    for( unsigned int mip = 0; mip < backingMipLevelDims.size(); ++mip )
        backingMipLevelDims[mip] = uint2{ std::max( backingInfo.width >> mip, 1U ), std::max( backingInfo.height >> mip, 1U ) };

    return m_backingImage->readMipTail( dest, backingMipTailFirstLevel, backingInfo.numMipLevels, &backingMipLevelDims[0], pixelSizeInBytes, stream );
}

}  // namespace imageSource
