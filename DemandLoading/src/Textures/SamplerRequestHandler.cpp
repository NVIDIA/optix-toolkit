//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "Textures/SamplerRequestHandler.h"
#include "DemandLoaderImpl.h"
#include "PagingSystem.h"
#include "Textures/DemandTextureImpl.h"
#include "Util/NVTXProfiling.h"

#include "TransferBufferDesc.h"

#include <algorithm>

#include <DemandLoading/Paging.h>  // for NON_EVICTABLE_LRU_VAL

namespace demandLoading {

void SamplerRequestHandler::fillRequest( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // We use MutexArray to ensure mutual exclusion on a per-page basis.  This is necessary because
    // multiple streams might race to create the same sampler.
    unsigned int index = pageId - m_startPage;
    MutexArrayLock lock( m_mutex.get(), index);

    // Do nothing if the request has already been filled.
    if( m_loader->getPagingSystem( deviceIndex )->isResident( pageId ) )
        return;

    // The samplers were the first resource that were assigned page table entries (via
    // PageTableManager), so the samplers occupy the first N page table entries.  The device code in
    // Texture2D.h relies on this invariant, but this code does not.
    unsigned int       samplerId = pageId - m_startPage;
    DemandTextureImpl* texture   = m_loader->getTexture( samplerId );

    // A 1x1 or null texture is indicated in the page table as a null value.
    imageSource::TextureInfo texInfo = {0};
    if( texture )
        texture->getImageSource()->open( &texInfo );

    if( texInfo.width <= 1 && texInfo.height <= 1 )
    {
        m_loader->getPagingSystem( deviceIndex )->addMapping( pageId, NON_EVICTABLE_LRU_VAL, 0ULL );
        return;
    }

    // Initialize the texture, reading image info from file header on the first call and
    // creating a per-device CUDA texture object.
    try
    {
        texture->init( deviceIndex );
    }
    catch( const std::exception& e )
    {
        std::stringstream ss;
        ss << "ImageSource::init() failed: " << e.what() << ": " << __FILE__ << " (" << __LINE__ << ")";
        throw Exception(ss.str().c_str());
    }

    // For a dense texture, the whole thing has to be loaded, so load it now
    if ( texture->useSparseTexture() == false )
        fillDenseTexture( deviceIndex, stream, pageId );

    // Allocate sampler buffer in pinned memory.
    PinnedItemPool<TextureSampler>* pinnedSamplerPool = m_loader->getPinnedMemoryManager()->getPinnedSamplerPool();
    TextureSampler*                 pinnedSampler     = pinnedSamplerPool->allocate();

    // Copy the canonical sampler from the DemandTexture and set its CUDA texture object, which differs per device.
    *pinnedSampler         = texture->getSampler();
    pinnedSampler->texture = texture->getTextureObject( deviceIndex );

    // Allocate device memory for sampler.
    SamplerPool*    samplerPool = m_loader->getDeviceMemoryManager( deviceIndex )->getSamplerPool();
    TextureSampler* devSampler  = samplerPool->allocate();

    // Copy sampler to device memory.
    DEMAND_CUDA_CHECK( cudaMemcpyAsync( devSampler, pinnedSampler, sizeof( TextureSampler ), cudaMemcpyHostToDevice, stream ) );

    // Free the pinned memory buffer.  This doesn't immediately reclaim it: an event is recorded on
    // the stream, and the buffer isn't reused until all preceding operations are complete,
    // including the asynchronous memcpy issued by fillTile().
    pinnedSamplerPool->free( pinnedSampler, deviceIndex, stream );

    // Push mapping for sampler to update page table.
    m_loader->getPagingSystem( deviceIndex )->addMapping( pageId, NON_EVICTABLE_LRU_VAL, reinterpret_cast<unsigned long long>( devSampler ) );
}

void SamplerRequestHandler::fillDenseTexture( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    DemandTextureImpl* texture = m_loader->getTexture( pageId );
    const imageSource::TextureInfo& info = texture->getInfo();

    // Try to get transfer buffer
    size_t textureSizeInBytes = getTextureSizeInBytes( info );
    TransferBufferDesc transferBuffer =
        m_loader->allocateTransferBuffer( deviceIndex, texture->getImageSource()->getFillType(), textureSizeInBytes, stream );

    // Make a backup buffer on the host if the transfer buffer was unsuccessful
    size_t hostBufferSize = ( transferBuffer.size == 0 && transferBuffer.memoryType == CU_MEMORYTYPE_HOST ) ? textureSizeInBytes : 0;
    std::vector<char> hostBuffer( hostBufferSize );

    // Get the final data pointer
    char* dataPtr = ( transferBuffer.size > 0 ) ? transferBuffer.buffer : hostBuffer.data();
    size_t bufferSize = std::max( hostBuffer.size(), transferBuffer.size );
    DEMAND_ASSERT_MSG( dataPtr != nullptr, "Unable to allocate transfer buffer for dense textures." );

    // Read the texture data into the buffer
    if( info.numMipLevels == 1 && ( info.width > 1 || info.height > 1 ) )
        texture->readNonMipMappedData( dataPtr, bufferSize, stream );
    else 
        texture->readMipLevels( dataPtr, bufferSize, 0, stream );

    // Copy texture data from the buffer to the texture array on the device
    texture->fillDenseTexture( deviceIndex, stream, dataPtr, info.width, info.height, transferBuffer.size > 0 );
    if( transferBuffer.size > 0 )
    {
        m_loader->freeTransferBuffer( transferBuffer, stream );
    }
    else 
    {
        // fillDenseTexture uses an async copy, so synchronize the stream when using the backup pageable buffer.
        DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
        DEMAND_CUDA_CHECK( cuStreamSynchronize( stream ) );
    }
}

}  // namespace demandLoading
