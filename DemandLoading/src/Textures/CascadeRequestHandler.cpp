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

#include "Textures/CascadeRequestHandler.h"

#include "DemandLoaderImpl.h"
#include "Memory/DeviceMemoryManager.h"
#include "PagingSystem.h"
#include "Textures/DemandTextureImpl.h"
#include "TransferBufferDesc.h"
#include "Util/NVTXProfiling.h"

#include <OptiXToolkit/DemandLoading/LRU.h>
#include <OptiXToolkit/DemandLoading/TextureSampler.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/CascadeImage.h>

#include <algorithm>

using namespace otk;

namespace demandLoading {

void CascadeRequestHandler::fillRequest( CUstream stream, unsigned int pageId )
{
    loadPage( stream, pageId, false );
}

void CascadeRequestHandler::loadPage( CUstream stream, unsigned int pageId, bool reloadIfResident )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // Use MutexArray to ensure mutual exclusion on a per-texture basis.
    unsigned int lockId = ( pageId - m_startPage ) / NUM_CASCADES;
    MutexArrayLock lock( m_mutex.get(), lockId );

    if( !reloadIfResident && m_loader->getPagingSystem()->isResident( pageId ) )
        return;

    unsigned int samplerId = cascadeIdToSamplerId( pageId );
    unsigned int requestCascadeSize = cascadeLevelToTextureSize( cascadeIdToCascadeLevel( pageId ) );

    DemandTextureImpl* texture = m_loader->getTexture( samplerId );
    imageSource::CascadeImage* cascadeImage = reinterpret_cast<imageSource::CascadeImage*>( texture->getImage().get() );
    std::shared_ptr<imageSource::ImageSource> backingImage = cascadeImage->getBackingImage();

    // Cascade image is already as large as possible, or larger than the proposed size. Just return
    const imageSource::TextureInfo& cascadeInfo = cascadeImage->getInfo();
    const imageSource::TextureInfo& backingInfo = backingImage->getInfo();
    if( cascadeInfo.width >= backingInfo.width || ( cascadeInfo.width >= requestCascadeSize && cascadeInfo.height >= requestCascadeSize ) )
        return;

    // Create a new cascadeImage and replace the current image with it.
    cascadeImage->setBackingImage( std::shared_ptr<imageSource::ImageSource>(nullptr) );
    unsigned int newCascadeSize = requestCascadeSize;
    std::shared_ptr<imageSource::ImageSource> newCascadeImage( new imageSource::CascadeImage( backingImage, newCascadeSize ) );
    m_loader->replaceTexture( stream, samplerId, newCascadeImage, texture->getDescriptor(), true );

    // Note: updating the page table is not necessary
    //m_loader->setPageTableEntry( pageId, false, 0ULL );
}

unsigned int CascadeRequestHandler::cascadeIdToSamplerId( unsigned int pageId )
{ 
    return ( pageId - m_startPage ) / NUM_CASCADES;
}

unsigned int CascadeRequestHandler::cascadeIdToCascadeLevel( unsigned int pageId )
{
    return ( pageId - m_startPage ) % NUM_CASCADES;
}

unsigned int CascadeRequestHandler::cascadeLevelToTextureSize( unsigned int cascadeLevel )
{
    const unsigned int MAX_TEXTURE_SIZE = 65536;
    return ( cascadeLevel >= NUM_CASCADES - 1 ) ? MAX_TEXTURE_SIZE : CASCADE_BASE << cascadeLevel;
}

}  // namespace demandLoading
