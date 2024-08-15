// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "RequestHandler.h"

namespace demandLoading {

class DemandLoaderImpl;
class DemandTextureImpl;

class SamplerRequestHandler : public RequestHandler
{
  public:
    /// Construct SamplerRequestHandler, which shares state with the DemandLoader.
    SamplerRequestHandler( DemandLoaderImpl* loader )
        : m_loader( loader )
    {
    }

    /// Fill a request for the specified page using the given stream.  
    void fillRequest( CUstream stream, unsigned int pageId ) override;

    /// Load or reload a page on the given stream
    void loadPage( CUstream stream, unsigned int pageId, bool reloadIfResident = true );

  private:
    bool fillDenseTexture( CUstream stream, unsigned int pageId );
    void fillBaseColorRequest( CUstream stream, DemandTextureImpl* texture, unsigned int pageId );

    DemandLoaderImpl* m_loader;
};

}  // namespace demandLoading
