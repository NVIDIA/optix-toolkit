// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Textures/SamplerRequestHandler.h"

#include "DemandLoaderImpl.h"
#include "Memory/DeviceMemoryManager.h"
#include "PagingSystem.h"
#include "Textures/DemandTextureImpl.h"
#include "TransferBufferDesc.h"
#include "Util/NVTXProfiling.h"

#include <OptiXToolkit/DemandLoading/DemandLoadLogger.h>
#include <OptiXToolkit/DemandLoading/LRU.h>
#include <OptiXToolkit/DemandLoading/TextureSampler.h>
#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/ImageSource/ImageHelpers.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <sstream>
#include <string>

using namespace otk;
using namespace imageSource;

namespace demandLoading {

void SamplerRequestHandler::fillRequest( CUstream stream, unsigned int pageId )
{
    try
    {
        loadPage( stream, pageId, false );
    }
    catch( const std::runtime_error& )
    {
        // Already recorded by throwRequestError. Leave the page non-resident so the request can be
        // retried, and let the worker notify the ticket.
    }
}

void SamplerRequestHandler::loadPage( CUstream stream, unsigned int pageId, bool reloadIfResident )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // We use MutexArray to ensure mutual exclusion on a per-page basis.  This is necessary because
    // multiple streams might race to create the same sampler.

    MutexArrayLock lock( m_mutex.get(), pageId - m_startPage );

    if( !reloadIfResident && m_loader->getPagingSystem()->isResident( pageId ) )
        return;

    // Get the texture and make sure it is open.
    unsigned int samplerId = pageIdToSamplerId( pageId, m_loader->getOptions().maxTextures );
    DemandTextureImpl* texture = nullptr;
    try
    {
        texture = m_loader->getTexture( samplerId );
    }
    catch( const std::exception& bang )
    {
        m_loader->throwRequestError( RequestErrorStage::Other, samplerId, nullptr,
                                     "Couldn't get texture for sampler id " + std::to_string( samplerId ) + " ("
                                         + bang.what() + ')' );
    }
    catch( ... )
    {
        m_loader->throwRequestError( RequestErrorStage::Other, samplerId, nullptr,
                                     "Couldn't get texture for sampler id " + std::to_string( samplerId ) );
    }

    try
    {
        texture->open();
    }
    catch( const std::exception& e )
    {
        m_loader->throwRequestError( RequestErrorStage::ImageLoad, texture->getId(), texture->getImage().get(),
                                     std::string( "ImageSource::open() failed: " ) + e.what() );
    }

    // Load base color if the page is for a base color
    if( isBaseColorId( pageId, m_loader->getOptions().maxTextures ) )
    {
        DL_LOG(4, "[Page " + std::to_string(pageId) + "] Base color request, texture " + std::to_string(samplerId) + ".");
        fillBaseColorRequest( stream, texture, pageId );
        return;
    }

    // If the texture is 1x1 or null, don't create a sampler.
    if( texture->isDegenerate() && !texture->isUdimEntryPoint() )
    {
        DL_LOG(4, "[Page " + std::to_string(pageId) + "] Degenerate texture " + std::to_string(samplerId) + ".");
        m_loader->setPageTableEntry( pageId, false, 0ULL );
        return;
    }

    // Initialize the texture, creating per-device CUDA texture objects.
    try
    {
        texture->init();
    }
    catch( const std::exception& e )
    {
        std::stringstream ss;
        ss << "ImageSource::init() failed: " << e.what() << ": " << __FILE__ << " (" << __LINE__ << ")";
        m_loader->throwRequestError( RequestErrorStage::SamplerCreate, texture->getId(), texture->getImage().get(), ss.str() );
    }

    DL_LOG(4, "[Page " + std::to_string(pageId) + "] Texture " + std::to_string(samplerId)
        + " (size=" + std::to_string(texture->getInfo().width) + "x" + std::to_string(texture->getInfo().height)
        + ", mips=" + std::to_string(texture->getInfo().numMipLevels)
        + ( !texture->useSparseTexture() ? ", dense" :
            ", sparse, tileSize=" + std::to_string(texture->getTileWidth()) + "x" + std::to_string(texture->getTileHeight()) )
        + ")");

    // For a dense texture, the whole thing has to be loaded, so load it now
    if ( !texture->useSparseTexture() )
    {
        // If the dense texture data was deferred, then defer allocating the sampler.
        if( !fillDenseTexture( stream, pageId ) )
            return;
    }

    // Allocate sampler buffer in pinned memory.
    MemoryBlockDesc pinnedBlock = m_loader->getPinnedMemoryPool()->alloc( sizeof( TextureSampler ), alignof( TextureSampler ) );
    if( pinnedBlock.isBad() )
    {
        m_loader->throwRequestError( RequestErrorStage::SamplerCreate, texture->getId(), texture->getImage().get(),
                                     "Failed to allocate pinned memory for sampler " + std::to_string( texture->getId() ) );
    }
    TextureSampler* pinnedSampler = reinterpret_cast<TextureSampler*>( pinnedBlock.ptr );

    // Copy the canonical sampler from the DemandTexture and set its CUDA texture object, which differs per device.
    *pinnedSampler         = texture->getSampler();
    pinnedSampler->texture = texture->getTextureObject();

    // Allocate device memory for device-side sampler.
    TextureSampler* devSampler = m_loader->getDeviceMemoryManager()->allocateSampler();
    if( reinterpret_cast<uint64_t>( devSampler ) == BAD_ADDR )
    {
        m_loader->getPinnedMemoryPool()->free( pinnedBlock );
        m_loader->throwRequestError( RequestErrorStage::SamplerCreate, texture->getId(), texture->getImage().get(),
                                     "Failed to allocate device memory for sampler " + std::to_string( texture->getId() ) );
    }

    // Copy sampler to device memory.
    try
    {
        OTK_ERROR_CHECK( cuMemcpyAsync( reinterpret_cast<CUdeviceptr>( devSampler ),
                                        reinterpret_cast<CUdeviceptr>( pinnedSampler ), sizeof( TextureSampler ), stream ) );
    }
    catch( const std::exception& e )
    {
        m_loader->getPinnedMemoryPool()->free( pinnedBlock );
        m_loader->throwRequestError( RequestErrorStage::SamplerCreate, texture->getId(), texture->getImage().get(),
                                     std::string( "Sampler copy failed: " ) + e.what() );
    }

    // Free the pinned memory buffer.  This doesn't immediately reclaim it: an event is recorded on
    // the stream, and the buffer isn't reused until all preceding operations are complete,
    // including the asynchronous memcpy issued by fillTile().
    m_loader->getPinnedMemoryPool()->freeAsync( pinnedBlock, stream );

    // Push mapping for sampler to update page table.
    m_loader->setPageTableEntry( pageId, false, reinterpret_cast<unsigned long long>( devSampler ) );
}

bool SamplerRequestHandler::fillDenseTexture( CUstream stream, unsigned int pageId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    DemandTextureImpl* texture = m_loader->getTexture( pageId );
    const imageSource::TextureInfo& info = texture->getInfo();

    // Try to get transfer buffer from the demand loader. We prefer it because it allows asynchronous fill.
    // The buffer needs to be a little larger than the texture size for some reason to prevent a crash, hence the extra 4 / 3.
    size_t transferBufferSize = getTextureSizeInBytes( info ) * 4 / 3;
    TransferBufferDesc transferBuffer =
        m_loader->allocateTransferBuffer( texture->getFillType(), transferBufferSize, stream );
    char* dataPtr = reinterpret_cast<char*>( transferBuffer.memoryBlock.ptr );
    size_t bufferSize = transferBuffer.memoryBlock.size;

    // Make a alternate buffer on the host if needed
    std::vector<char> hostBuffer;
    if( transferBuffer.memoryType == CU_MEMORYTYPE_HOST && transferBuffer.memoryBlock.size == 0 )
    {
        hostBuffer.resize( transferBufferSize );
        dataPtr = hostBuffer.data();
        bufferSize = transferBufferSize;
    }

    // Make alternate buffer on device if needed
    if( transferBuffer.memoryType == CU_MEMORYTYPE_DEVICE && transferBuffer.memoryBlock.size == 0 )
    {
        OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &dataPtr ), transferBufferSize ) );
        bufferSize = transferBufferSize;
    }

    OTK_ASSERT_MSG( dataPtr != nullptr, "Unable to allocate transfer buffer for dense texture." );

    // Read the texture data into the buffer (either a single mip level, or all mip levels)
    bool satisfied = false;
    try
    {
        if( info.numMipLevels == 1 && !texture->isDegenerate() )
            satisfied = texture->readNonMipMappedData( dataPtr, bufferSize, stream );
        else
            satisfied = texture->readMipLevels( dataPtr, bufferSize, 0, stream );
    }
    catch( const std::exception& e )
    {
        if( transferBuffer.memoryBlock.size > 0 )
            m_loader->freeTransferBuffer( transferBuffer, stream );
        else if( transferBuffer.memoryType == CU_MEMORYTYPE_DEVICE )
            cuMemFree( reinterpret_cast<CUdeviceptr>( dataPtr ) );

        std::stringstream ss;
        ss << "Image read failed: " << e.what() << ": " << __FILE__ << " (" << __LINE__ << ")";
        m_loader->throwRequestError( RequestErrorStage::ImageLoad, texture->getId(), texture->getImage().get(), ss.str() );
    }

    // Copy texture data from the buffer to the texture array on the device
    if( satisfied )
        texture->fillDenseTexture( stream, dataPtr, info.width, info.height, transferBuffer.memoryBlock.size > 0 );

    // Free the transfer buffer from the demand loader
    if( transferBuffer.memoryBlock.size > 0 )
    {
        m_loader->freeTransferBuffer( transferBuffer, stream );
    }
    else 
    {
        // fillDenseTexture uses an async copy, so synchronize the stream when using an alternate buffer.
        OTK_ERROR_CHECK( cuStreamSynchronize( stream ) );

        // Free device-side alternate buffer if one was made
        if( transferBuffer.memoryType == CU_MEMORYTYPE_DEVICE )
            cuMemFree( reinterpret_cast<CUdeviceptr>( dataPtr ) );
    }

    return satisfied;
}

void SamplerRequestHandler::fillBaseColorRequest( CUstream /*stream*/, DemandTextureImpl* texture, unsigned int pageId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // Read the base color
    float4 fBaseColor = float4{1.0f, 0.0f, 1.0f, 0.0f};
    bool hasBaseColor = false;
    try
    {
        hasBaseColor = texture->readBaseColor( fBaseColor );
    }
    catch( const std::exception& e )
    {
        m_loader->throwRequestError( RequestErrorStage::ImageLoad, texture->getId(), texture->getImage().get(),
                                     std::string( "Base color read failed: " ) + e.what() );
    }

    // Store the base color as a half4 in the page table
    half4               baseColor = half4{fBaseColor.x, fBaseColor.y, fBaseColor.z, fBaseColor.w};
    unsigned long long* baseVal   = reinterpret_cast<unsigned long long*>( &baseColor );
    unsigned long long  val = ( hasBaseColor ) ? *baseVal : NO_BASE_COLOR;
    m_loader->getPagingSystem()->addMapping( pageId, NON_EVICTABLE_LRU_VAL, val );
}

}  // namespace demandLoading
