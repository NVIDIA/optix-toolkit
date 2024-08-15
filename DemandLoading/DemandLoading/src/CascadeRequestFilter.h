// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <stdio.h>
#include <OptiXToolkit/DemandLoading/RequestFilter.h>
#include "DemandLoaderImpl.h"

namespace demandLoading {

class CascadeRequestFilter : public RequestFilter
{
  public:
    CascadeRequestFilter( unsigned int cascadePagesStart, unsigned int cascadePagesEnd, DemandLoaderImpl* demandLoader )
      : m_cascadePagesStart( cascadePagesStart )
      , m_cascadePagesEnd( cascadePagesEnd )
      , m_demandLoader( demandLoader )
    {
    }
    std::vector<unsigned int> filter( const unsigned int* requests, unsigned int numRequests ) override;

  private:
    unsigned int m_cascadePagesStart;
    unsigned int m_cascadePagesEnd;
    DemandLoaderImpl* m_demandLoader;

    bool isCascadePage( unsigned int pageId ) 
    { 
        return pageId >= m_cascadePagesStart && pageId < m_cascadePagesEnd;
    }

    unsigned int cascadePageToTextureId( unsigned int pageId )
    {
        return ( pageId - m_cascadePagesStart ) / NUM_CASCADES;
    }
};

}  // namespace demandLoading
