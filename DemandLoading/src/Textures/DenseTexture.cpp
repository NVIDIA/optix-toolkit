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

#include "Textures/DenseTexture.h"
#include "Util/ContextSaver.h"

#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>

#include <vector_functions.h> // from CUDA toolkit

#include <algorithm>
#include <cmath>

namespace demandLoading {

void DenseTexture::init( const TextureDescriptor& descriptor, const imageSource::TextureInfo& info, std::shared_ptr<CUmipmappedArray> masterArray )
{
    // Redundant initialization can occur since requests from multiple streams are not deduplicated.
    if( m_isInitialized && info == m_info )
        return;

    // Record the current CUDA context.
    OTK_ERROR_CHECK( cuCtxGetCurrent( &m_context ) );

    m_info = info;

    // Create CUDA array, or use masterArray if one was provided
    CUDA_ARRAY3D_DESCRIPTOR ad{};
    ad.Width       = m_info.width;
    ad.Height      = m_info.height;
    ad.Format      = m_info.format;
    ad.NumChannels = m_info.numChannels;

    m_array = masterArray;
    if( m_array.get() == nullptr )
    {
        CUmipmappedArray* array = new CUmipmappedArray();
        OTK_ERROR_CHECK( cuMipmappedArrayCreate( array, &ad, m_info.numMipLevels ) );

        // Reset m_array with a deleter for the mipmapped array
        m_array.reset( array, [this]( CUmipmappedArray* array ) {
            OTK_ERROR_CHECK_NOTHROW( cuCtxPushCurrent( m_context ) );
            OTK_ERROR_CHECK_NOTHROW( cuMipmappedArrayDestroy( *array ) );
            delete array;
            CUcontext ignored;
            OTK_ERROR_CHECK_NOTHROW( cuCtxPopCurrent( &ignored ) );
        } );
    }

    // Create CUDA texture descriptor
    CUDA_TEXTURE_DESC td{};
    td.addressMode[0]      = descriptor.addressMode[0];
    td.addressMode[1]      = descriptor.addressMode[1];
    td.filterMode          = descriptor.filterMode;
    td.flags               = CU_TRSF_NORMALIZED_COORDINATES | descriptor.flags;
    td.maxAnisotropy       = descriptor.maxAnisotropy;
    td.mipmapFilterMode    = descriptor.mipmapFilterMode;
    td.maxMipmapLevelClamp = float( info.numMipLevels - 1 );
    td.minMipmapLevelClamp = 0.f;

    // Create texture object.
    CUDA_RESOURCE_DESC rd{};
    rd.resType                    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    rd.res.mipmap.hMipmappedArray = *m_array;
    OTK_ERROR_CHECK( cuTexObjectCreate( &m_texture, &rd, &td, nullptr ) );

    m_isInitialized = true;
}

uint2 DenseTexture::getMipLevelDims( unsigned int mipLevel ) const
{
    // Get CUDA array for the specified level from the mipmapped array.
    OTK_ASSERT( mipLevel < m_info.numMipLevels );
    CUarray mipLevelArray; 
    OTK_ERROR_CHECK( cuMipmappedArrayGetLevel( &mipLevelArray, *m_array, mipLevel ) );

    // Get the array descriptor.
    CUDA_ARRAY_DESCRIPTOR desc;
    OTK_ERROR_CHECK( cuArrayGetDescriptor( &desc, mipLevelArray ) );

    return make_uint2( static_cast<unsigned int>( desc.Width ), static_cast<unsigned int>( desc.Height ) );
}

void DenseTexture::fillTexture( CUstream stream, const char* textureData, unsigned int width, unsigned int height, bool bufferPinned ) const
{
    OTK_ASSERT( m_isInitialized );
    OTK_ASSERT( width == m_info.width );
    OTK_ASSERT( height == m_info.height );
    (void)width; // silence unused variable warning
    (void)height;

    // Fill each level.
    size_t             offset    = 0;
    const unsigned int pixelSize = m_info.numChannels * imageSource::getBytesPerChannel( m_info.format );

    for( unsigned int mipLevel = 0; mipLevel < m_info.numMipLevels; ++mipLevel )
    {
        CUarray mipLevelArray{};
        OTK_ERROR_CHECK( cuMipmappedArrayGetLevel( &mipLevelArray, *m_array, mipLevel ) );

        uint2 levelDims = getMipLevelDims( mipLevel );

        CUDA_MEMCPY2D copyArgs{};
        copyArgs.srcMemoryType = CU_MEMORYTYPE_HOST;
        copyArgs.srcHost       = textureData + offset;
        copyArgs.srcPitch      = levelDims.x * pixelSize;

        copyArgs.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copyArgs.dstArray      = mipLevelArray;

        copyArgs.WidthInBytes = levelDims.x * pixelSize;
        copyArgs.Height       = levelDims.y;

        if( bufferPinned )
            OTK_ERROR_CHECK( cuMemcpy2DAsync( &copyArgs, stream ) );
        else 
            OTK_ERROR_CHECK( cuMemcpy2D( &copyArgs ) );

        offset += levelDims.x * levelDims.y * pixelSize;
        m_numBytesFilled += copyArgs.WidthInBytes * copyArgs.Height;
    }
}

DenseTexture::~DenseTexture()
{
    if( m_isInitialized )
    {
        // m_array destroyed by shared_ptr deleter
        m_array.reset();
        ContextSaver contextSaver;
        OTK_ERROR_CHECK_NOTHROW( cuCtxSetCurrent( m_context ) );
        OTK_ERROR_CHECK_NOTHROW( cuTexObjectDestroy( m_texture ) );
    }
}

}  // namespace demandLoading
