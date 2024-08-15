// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "RequestHandler.h"
#include <OptiXToolkit/Memory/MemoryBlockDesc.h>

#include <atomic>

namespace demandLoading {

class DemandLoaderImpl;
class DemandTextureImpl;

class TextureRequestHandler : public RequestHandler
{
  public:
    /// Default constructor.
    TextureRequestHandler() {}

    /// Construct TextureRequestHandler, which shares state with the DemandLoader.
    TextureRequestHandler( DemandTextureImpl* texture, DemandLoaderImpl* loader )
        : m_texture( texture )
        , m_loader( loader )
    {
    }

    /// Fill a request for the specified page using the given stream.  
    void fillRequest( CUstream stream, unsigned int pageId ) override;

    // Load or reload a page
    void loadPage( CUstream stream, unsigned int pageId, bool reloadIfResident );

    /// Get the associated texture.
    DemandTextureImpl* getTexture() const { return m_texture; }

    /// Unmap the backing storage associated with a texture tile or mip tail
    void unmapTileResource( CUstream stream, unsigned int pageId );

    /// Get the pageId for a tile
    unsigned int getTextureTilePageId( unsigned int mipLevel, unsigned int tileX, unsigned int tileY );

  private:
    DemandTextureImpl* m_texture = nullptr;
    DemandLoaderImpl*  m_loader = nullptr;

    void fillTileRequest( CUstream stream, unsigned int pageId, otk::TileBlockHandle bh );
    void fillMipTailRequest( CUstream stream, unsigned int pageId, otk::TileBlockHandle bh );
};

}  // namespace demandLoading
