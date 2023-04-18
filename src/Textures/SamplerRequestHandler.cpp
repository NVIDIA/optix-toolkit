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

#include "Textures/SamplerRequestHandler.h"

#include "DemandLoaderImpl.h"
#include "Memory/DeviceMemoryManager.h"
#include "PagingSystem.h"
#include "Textures/DemandTextureImpl.h"
#include "TransferBufferDesc.h"
#include "Util/NVTXProfiling.h"

#include <OptiXToolkit/DemandLoading/Paging.h>  // for NON_EVICTABLE_LRU_VAL

#include <cuda_fp16.h>

#include <algorithm>

using namespace otk;

namespace demandLoading {

void SamplerRequestHandler::fillRequest( CUstream stream, unsigned int pageId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // We use MutexArray to ensure mutual exclusion on a per-page basis.  This is necessary because
    // multiple streams might race to create the same sampler.
    MutexArrayLock lock( m_mutex.get(), pageId - m_startPage );

    // Do nothing if the request has already been filled.
    PagingSystem* pagingSystem = m_loader->getPagingSystem();
    if( pagingSystem->isResident( pageId ) )
        return;

    // Get the texture and make sure it is open.
    unsigned int samplerId = pageIdToSamplerId( pageId );
    DemandTextureImpl* texture = m_loader->getTexture( samplerId );

    texture->open();

    // Load base color if the page is for a base color
    if( isBaseColorId( pageId ) )
    {
        fillBaseColorRequest( stream, texture, pageId );
        return;
    }

    // If the texture is 1x1 or null, don't create a sampler.
    if( texture->isDegenerate() && !texture->isUdimEntryPoint() )
    {
        m_loader->setPageTableEntry( pageId, false, nullptr );
        return;
    }

    // Initialize the texture, creating a per-device CUDA texture objects.
    try
    {
        texture->init();
    }
    catch( const std::exception& e )
    {
        std::stringstream ss;
        ss << "ImageSource::init() failed: " << e.what() << ": " << __FILE__ << " (" << __LINE__ << ")";
        throw Exception(ss.str().c_str());
    }

    // For a dense texture, the whole thing has to be loaded, so load it now
    if ( !texture->useSparseTexture() )
    {
        // If the dense texture data was deferred, then defer allocating the sampler.
        if( !fillDenseTexture( stream, pageId ) )
            return;
    }

    // Allocate sampler buffer in pinned memory.
    MemoryBlockDesc pinnedBlock = m_loader->getPinnedMemoryPool()->alloc( sizeof( TextureSampler ), alignof( TextureSampler ) );
    TextureSampler* pinnedSampler = reinterpret_cast<TextureSampler*>( pinnedBlock.ptr );

    // Copy the canonical sampler from the DemandTexture and set its CUDA texture object, which differs per device.
    *pinnedSampler         = texture->getSampler();
    pinnedSampler->texture = texture->getTextureObject();

    // Allocate device memory for device-side sampler.
    TextureSampler* devSampler = m_loader->getDeviceMemoryManager()->allocateSampler();

    // Copy sampler to device memory.
    DEMAND_CUDA_CHECK( cuMemcpyAsync( reinterpret_cast<CUdeviceptr>( devSampler ),
                                      reinterpret_cast<CUdeviceptr>( pinnedSampler ), sizeof( TextureSampler ), stream ) );

    // Free the pinned memory buffer.  This doesn't immediately reclaim it: an event is recorded on
    // the stream, and the buffer isn't reused until all preceding operations are complete,
    // including the asynchronous memcpy issued by fillTile().
    m_loader->getPinnedMemoryPool()->freeAsync( pinnedBlock, stream );

    // Push mapping for sampler to update page table.
    m_loader->setPageTableEntry( pageId, false, devSampler );
}

bool SamplerRequestHandler::fillDenseTexture( CUstream stream, unsigned int pageId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    DemandTextureImpl* texture = m_loader->getTexture( pageId );
    const imageSource::TextureInfo& info = texture->getInfo();

    // Try to get transfer buffer
    // The buffer needs to be a little larger than the texture size for some reason to prevent a crash
    size_t transferBufferSize = getTextureSizeInBytes( info ) * 4 / 3;
    TransferBufferDesc transferBuffer =
        m_loader->allocateTransferBuffer( texture->getFillType(), transferBufferSize, stream );

    // Make a backup buffer on the host if the transfer buffer was unsuccessful
    size_t hostBufferSize = ( transferBuffer.memoryBlock.size == 0 && transferBuffer.memoryType == CU_MEMORYTYPE_HOST ) ? transferBufferSize : 0;
    std::vector<char> hostBuffer( hostBufferSize );

    // Get the final data pointer
    char* dataPtr = ( transferBuffer.memoryBlock.size > 0 ) ? reinterpret_cast<char*>( transferBuffer.memoryBlock.ptr ) : hostBuffer.data();
    size_t bufferSize = std::max( hostBuffer.size(), transferBuffer.memoryBlock.size );
    DEMAND_ASSERT_MSG( dataPtr != nullptr, "Unable to allocate transfer buffer for dense textures." );

    // Read the texture data into the buffer
    bool satisfied;
    if( info.numMipLevels == 1 && !texture->isDegenerate() )
        satisfied = texture->readNonMipMappedData( dataPtr, bufferSize, stream );
    else
        satisfied = texture->readMipLevels( dataPtr, bufferSize, 0, stream );

    // Copy texture data from the buffer to the texture array on the device
    if( satisfied )
        texture->fillDenseTexture( stream, dataPtr, info.width, info.height, transferBuffer.memoryBlock.size > 0 );
    if( transferBuffer.memoryBlock.size > 0 )
    {
        m_loader->freeTransferBuffer( transferBuffer, stream );
    }
    else 
    {
        // fillDenseTexture uses an async copy, so synchronize the stream when using the backup pageable buffer.
        DEMAND_CUDA_CHECK( cuStreamSynchronize( stream ) );
    }

    return satisfied;
}

struct half4
{
    half x, y, z, w;
};

void SamplerRequestHandler::fillBaseColorRequest( CUstream stream, DemandTextureImpl* texture, unsigned int pageId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // Read the base color
    float4 fBaseColor = float4{1.0f, 0.0f, 1.0f, 0.0f};
    bool hasBaseColor = false;
    hasBaseColor = texture->readBaseColor( fBaseColor );

    // Store the base color as a half4 in the page table
    unsigned long long  noColor   = 0xFFFFFFFFFFFFFFFFull; // four half NaNs, to indicate when no baseColor exists
    half4               baseColor = half4{fBaseColor.x, fBaseColor.y, fBaseColor.z, fBaseColor.w};
    unsigned long long* baseVal   = ( hasBaseColor ) ? reinterpret_cast<unsigned long long*>( &baseColor ) : &noColor;
    m_loader->getPagingSystem()->addMapping( pageId, NON_EVICTABLE_LRU_VAL, *baseVal );
}

}  // namespace demandLoading
