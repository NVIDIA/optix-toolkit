// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Textures/CascadeRequestHandler.h"

#include "DemandLoaderImpl.h"
#include "Memory/DeviceMemoryManager.h"
#include "PagingSystem.h"
#include "Textures/DemandTextureImpl.h"
#include "TransferBufferDesc.h"
#include "Util/NVTXProfiling.h"

#include <OptiXToolkit/DemandLoading/DemandLoadLogger.h>
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

    DL_LOG(4, "[Page " + std::to_string(pageId) + "] Cascade request, texture " + std::to_string(samplerId)
        + ", cascade size " + std::to_string(requestCascadeSize) + ".");

    DemandTextureImpl* texture = m_loader->getTexture( samplerId );
    imageSource::CascadeImage* cascadeImage = reinterpret_cast<imageSource::CascadeImage*>( texture->getImage().get() );
    std::shared_ptr<imageSource::ImageSource> backingImage = cascadeImage->getBackingImage();

    // If the cascade image is already as large as possible, or larger than the proposed size, return.
    const imageSource::TextureInfo& cascadeInfo = cascadeImage->getInfo();
    const imageSource::TextureInfo& backingInfo = backingImage->getInfo();
    if( cascadeInfo.width >= backingInfo.width || ( cascadeInfo.width >= requestCascadeSize && cascadeInfo.height >= requestCascadeSize ) )
        return;

    // Clear the backing image and page table entry on the device.
    cascadeImage->setBackingImage( std::shared_ptr<imageSource::ImageSource>(nullptr) );
    m_loader->setPageTableEntry( pageId, false, 0ULL );

    // Create a new cascadeImage and replace the current image with it.
    unsigned int newCascadeSize = requestCascadeSize;
    std::shared_ptr<imageSource::ImageSource> newCascadeImage( new imageSource::CascadeImage( backingImage, newCascadeSize ) );
    m_loader->replaceTexture( stream, samplerId, newCascadeImage, texture->getDescriptor(), true );
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
