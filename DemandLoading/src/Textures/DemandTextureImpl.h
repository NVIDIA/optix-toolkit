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
#pragma once

#include "Textures/DenseTexture.h"
#include "Textures/SparseTexture.h"
#include "Textures/TextureRequestHandler.h"
#include "Util/Exception.h"
#include "Util/PerContextData.h"

#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>
#include <OptiXToolkit/DemandLoading/TextureSampler.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>

#include <cuda.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace imageSource {
class ImageSource;
}

namespace demandLoading {

class DemandLoaderImpl;
struct Statistics;
class TilePool;

/// Demand-loaded textures are created by the DemandLoader.
class DemandTextureImpl : public DemandTexture
{
  public:
    /// Default constructor.
    DemandTextureImpl() = default;

    /// Construct demand loaded texture with the specified id (which is used as an index into the
    /// device-side sampler array) the the given descriptor (which specifies the wrap mode, filter
    /// mode, etc.).  The given image reader is retained and used by subsequent readTile() calls.
    DemandTextureImpl( unsigned int                              id,
                       const TextureDescriptor&                  descriptor,
                       std::shared_ptr<imageSource::ImageSource> image,
                       DemandLoaderImpl*                         loader );

    /// Construct a variant demand loaded texture based on mainTexture.  The variant will use the same
    /// sparse texture backing store as mainTexture, so texture tiles can be shared between the textures.
    DemandTextureImpl( unsigned int id, DemandTextureImpl* masterTexture, const TextureDescriptor& descriptor, DemandLoaderImpl* loader );

    /// Default destructor.
    ~DemandTextureImpl() override = default;

    /// DemandTextureImpl cannot be copied because the PageTableManager holds a pointer to the
    /// RequestHandler it provides.
    DemandTextureImpl( const DemandTextureImpl& ) = delete;

    /// Not assignable.
    DemandTextureImpl& operator=( const DemandTextureImpl& ) = delete;

    /// A degenerate texture is handled by a base color.
    bool isDegenerate() const { return m_info.width <= 1 && m_info.height <= 1; }

    /// Read the base color of the associated image.
    bool readBaseColor( float4& baseColor ) const { return m_image->readBaseColor( baseColor ); }

    /// Get the memory fill type for this texture.
    CUmemorytype getFillType() const { return m_image->getFillType(); }

    /// Replace the current texture image. Return true if the sampler for the texture needs to be updated.
    bool setImage( const TextureDescriptor& descriptor, std::shared_ptr<imageSource::ImageSource> newImage );

    /// Get the texture id, which is used as an index into the device-side sampler array.
    unsigned int getId() const override;

    /// Initialize the texture.  When first called, this method opens the image reader that was
    /// provided to the constructor.  Throws an exception on error.
    void init();

    /// Get the image info.  Valid only after the image has been initialized (e.g. opened).
    const imageSource::TextureInfo& getInfo() const;

    /// Get the canonical sampler for this texture, excluding the CUDA texture object, which differs
    /// for each device (see getTextureObject).
    const TextureSampler& getSampler() const;

    /// Get the CUDA texture object for the current CUDA context.
    CUtexObject getTextureObject() const;

    /// Get the texture descriptor
    const TextureDescriptor& getDescriptor() const;

    /// Get the dimensions of the specified miplevel.
    uint2 getMipLevelDims( unsigned int mipLevel ) const;

    /// Get tile width.
    unsigned int getTileWidth() const;

    /// Get tile height.
    unsigned int getTileHeight() const;

    /// Return whether the texture is mipmapped. 
    /// Throws an exception if m_info has not been initialized.
    bool isMipmapped() const;

    /// Return whether to use a sparse or dense texture. 
    /// Throws an exception if m_info has not been initialized.
    bool useSparseTexture() const;

    /// Get the first miplevel in the mip tail.
    unsigned int getMipTailFirstLevel() const;

    /// Get the request handler for this texture.
    TextureRequestHandler* getRequestHandler() { return m_requestHandler.get(); }

    /// Accumulate statistics for this texture, if the associated ImageSource is not in the set.
    void accumulateStatistics( Statistics& stats, std::set<imageSource::ImageSource*>& images );

    /// Read the specified tile into the given buffer.
    /// Throws an exception on error.
    bool readTile( unsigned int mipLevel, unsigned int tileX, unsigned int tileY, char* tileBuffer,
                   size_t tileBufferSize, CUstream stream ) const;

    /// Fill the device tile backing storage for a texture tile and with the given data.
    void fillTile( CUstream                     stream,
                   unsigned int                 mipLevel,
                   unsigned int                 tileX,
                   unsigned int                 tileY,
                   const char*                  tileData,
                   CUmemorytype                 tileDataType,
                   size_t                       tileSize,
                   CUmemGenericAllocationHandle handle,
                   size_t                       offset ) const;

    /// Unmap backing storage for a tile
    void unmapTile( CUstream stream, unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) const;

    /// Read the entire non-mipmapped texture into the buffer.
    /// Throws an exception on error.
    bool readNonMipMappedData( char* buffer, size_t bufferSize, CUstream stream ) const;

    /// Read all the levels in the mip tail into the given buffer.
    /// Throws an exception on error.
    bool readMipTail( char* buffer, size_t bufferSize, CUstream stream ) const;

    /// Read all the levels from startLevel into the given buffer.
    /// Throws an exception on error.
    bool readMipLevels( char* buffer, size_t bufferSize, unsigned int startLevel, CUstream stream ) const;

    /// Fill the device backing storage for the mip tail with the given data.
    void fillMipTail( CUstream                     stream,
                      const char*                  mipTailData,
                      CUmemorytype                 mipTailDataType,
                      size_t                       mipTailSize,
                      CUmemGenericAllocationHandle handle,
                      size_t                       offset ) const;

    /// Unmap backing storage for the mip tail
    void unmapMipTail( CUstream stream ) const;

    /// Create and fill the dense texture on the given device
    void fillDenseTexture( CUstream stream, const char* textureData, unsigned int width, unsigned int height, bool bufferPinned );

    /// Opens the corresponding ImageSource and obtains basic information about the texture dimensions.
    void open();

    bool isOpen() const { return m_isOpen; }

    /// Set this texture as an entry point to a udim texture array
    void setUdimTexture( unsigned int udimStartPage, unsigned int udim, unsigned int vdim, bool isBaseTexture );

    /// Return true if the texture is an entry point for a udim texture
    bool isUdimEntryPoint() { return ( m_sampler.udim > 0 ); }
    
    /// Return the size of the mip tail if the texture is initialized.
    size_t getMipTailSize(); 

  private:
    // A mutex guards against concurrent initialization, which can arise when the sampler
    // is requested on multiple devices.  Tiles can be filled concurrently, along with the mip tail.
    std::mutex m_initMutex;

    // The texture identifier is used as an index into the device-side sampler array.
    const unsigned int m_id = 0;

    // The texture descriptor specifies wrap and filtering modes, etc.  Invariant after init(), and not valid before then.
    TextureDescriptor m_descriptor{};

    // The image provides a read() method that fills requested miplevels.
    std::shared_ptr<imageSource::ImageSource> m_image;

    // Master texture, if this is a texture variant (shares image backing store with master texture). 
    DemandTextureImpl* m_masterTexture;

    // The DemandLoader provides access to the PageTableManager, etc.
    DemandLoaderImpl* const m_loader;

    // The image is lazily opened.  Invariant after open().
    bool m_isOpen{};

    // The texture is lazily initialized.  Invariant after init().
    bool m_isInitialized{};

    // Image info, including dimensions and format.  Invariant after init(), and not valid before then.
    imageSource::TextureInfo m_info{};
    TextureSampler     m_sampler{};
    unsigned int       m_tileWidth         = 0;
    unsigned int       m_tileHeight        = 0;
    unsigned int       m_mipTailFirstLevel = 0;
    size_t             m_mipTailSize       = 0;
    std::vector<uint2> m_mipLevelDims;

    // Sparse and dense textures (one per CUDA context).
    PerContextData<SparseTexture> m_sparseTextures;
    PerContextData<DenseTexture> m_denseTextures;
    std::mutex m_sparseTexturesMutex;
    std::mutex m_denseTexturesMutex;

    // Request handler.
    std::unique_ptr<TextureRequestHandler> m_requestHandler;

    void         initSampler();
    unsigned int getNumTilesInLevel( unsigned int mipLevel ) const;

    // Threshold number of pixels to switch between sparse and dense texture
    const unsigned int SPARSE_TEXTURE_THRESHOLD = 1024;

    // Get the sparse texture for the current CUDA context, creating it if necessary.
    SparseTexture& getSparseTexture();

    // Get a const reference to the sparse texture for the current CUDA context, creating it if necessary.
    const SparseTexture& getSparseTexture() const { return const_cast<DemandTextureImpl*>( this )->getSparseTexture(); }

    // Get the dense texture for the current CUDA context, creating it if necessary.
    DenseTexture& getDenseTexture();

    // Get a const reference to the dense texture for the current CUDA context, creating it if necessary.
    const DenseTexture& getDenseTexture() const { return const_cast<DemandTextureImpl*>( this )->getDenseTexture(); }
};

}  // namespace demandLoading
