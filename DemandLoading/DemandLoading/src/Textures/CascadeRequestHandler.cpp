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
    try
    {
        loadPage( stream, pageId, false );
    }
    catch( const std::runtime_error& e )
    {
        // Report the failure (e.g. the grown texture is missing or too large to sample) through the
        // logger instead of letting it propagate and terminate the worker thread.  Logged at level 0
        // so it is reported regardless of the configured log level.  loadPage leaves the cascade page
        // non-resident on failure, so the request will be retried.
        DL_LOG( 0, e.what() );
    }
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

    // Clear the backing image on the device.
    cascadeImage->setBackingImage( std::shared_ptr<imageSource::ImageSource>(nullptr) );

    // Create a new cascadeImage and replace the current image with it.
    unsigned int newCascadeSize = requestCascadeSize;
    std::shared_ptr<imageSource::ImageSource> newCascadeImage( new imageSource::CascadeImage( backingImage, newCascadeSize ) );
    m_loader->replaceTexture( stream, samplerId, newCascadeImage, texture->getDescriptor(), true );

    // Mark the cascade page resident only after the replacement succeeds.  If replaceTexture throws,
    // the page is left non-resident so the cascade request will be retried rather than silently
    // stranded as resident with a zero mapping.
    m_loader->setPageTableEntry( pageId, false, 0ULL );
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
