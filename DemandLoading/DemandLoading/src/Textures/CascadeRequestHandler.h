// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "RequestHandler.h"

namespace demandLoading {

class DemandLoaderImpl;

class CascadeRequestHandler : public RequestHandler
{
  public:
    /// Construct CascaderRequestHandler for the given DemandLoaderImpl.
    CascadeRequestHandler( DemandLoaderImpl* loader ) 
    : m_loader{ loader }
    {}

    /// Fill a request for the specified page on the stream.
    void fillRequest( CUstream stream, unsigned int pageId ) override;

    /// Load or reload a page on the given stream.
    void loadPage( CUstream stream, unsigned int pageId, bool reloadIfResident );

  private:
    DemandLoaderImpl* m_loader;

    unsigned int cascadeIdToSamplerId( unsigned int pageId );
    unsigned int cascadeIdToCascadeLevel( unsigned int pageId );
    unsigned int cascadeLevelToTextureSize( unsigned int cascadeLevel );
};

}  // namespace demandLoading
