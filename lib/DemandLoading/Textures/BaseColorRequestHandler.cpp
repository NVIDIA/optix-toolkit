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

#include "Textures/BaseColorRequestHandler.h"
#include "DemandLoaderImpl.h"
#include "PagingSystem.h"
#include "Textures/DemandTextureImpl.h"
#include "Util/NVTXProfiling.h"

#include <DemandLoading/Paging.h>  // for NON_EVICTABLE_LRU_VAL

#include <cuda_fp16.h>

namespace demandLoading {

struct half4
{
    half x, y, z, w;
};

void BaseColorRequestHandler::fillRequest( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    unsigned int       textureId = pageId - m_startPage;
    MutexArrayLock     lock( m_mutex.get(), textureId );
    DemandTextureImpl* texture = m_loader->getTexture( textureId );

    // Do nothing if the request has already been filled.
    if( m_loader->getPagingSystem( deviceIndex )->isResident( pageId ) )
        return;

    float4 fBaseColor = float4{1.0f, 0.0f, 1.0f, 0.0f};
    bool hasBaseColor = false;
    if( texture != nullptr )
    {
        texture->getImageSource()->open( nullptr );
        hasBaseColor = texture->getImageSource()->readBaseColor( fBaseColor );
    }

    // Store the base color as a half4 in the page table
    unsigned long long  noColor   = 0xFFFFFFFFFFFFFFFFull; // four half NaNs, to indicate when no baseColor exists
    half4               baseColor = half4{fBaseColor.x, fBaseColor.y, fBaseColor.z, fBaseColor.w};
    unsigned long long* baseVal   = ( hasBaseColor ) ? reinterpret_cast<unsigned long long*>( &baseColor ) : &noColor;
    m_loader->getPagingSystem( deviceIndex )->addMapping( pageId, NON_EVICTABLE_LRU_VAL, *baseVal );
}

}  // namespace demandLoading
