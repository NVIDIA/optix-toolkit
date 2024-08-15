// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "CascadeRequestFilter.h"

namespace demandLoading {

std::vector<unsigned int> CascadeRequestFilter::filter( const unsigned int* requests, unsigned int numRequests )
{
    // Gather and sort the cascade requests.
    std::vector<unsigned int> cascadePages;
    for( unsigned int i = 0; i < numRequests; ++i )
    {
        if( isCascadePage( requests[i] ) )
        {
            unsigned int request = requests[i];
            unsigned int textureId = cascadePageToTextureId(request);
            DemandTextureImpl* texture = m_demandLoader->getTexture( textureId );
            if( texture->getMasterTexture() != nullptr )
            {
                // Change request for a variant texture cascade to the corresponding master texture cascade.
                unsigned int cascadeNum = ( request - m_cascadePagesStart ) % NUM_CASCADES;
                request = m_cascadePagesStart + NUM_CASCADES * texture->getMasterTexture()->getId() + cascadeNum;
            }
            cascadePages.push_back( request );
        }
    }
    std::sort( cascadePages.begin(), cascadePages.end() );

    // Filter cascade requests to keep the largest cascade for each texture
    std::vector<unsigned int> filteredRequests;
    for( int i = static_cast<int>( cascadePages.size() - 1 ); i >= 0 ; --i)
    {
        if( filteredRequests.empty() || ( cascadePageToTextureId( cascadePages[i] ) != cascadePageToTextureId( cascadePages[i + 1] ) ) )
            filteredRequests.push_back( cascadePages[i] );
    }

    // Find current page range for each cascade's texture, put in map so we can remove these requests
    std::map<unsigned int, unsigned int> knockoutPages;
    knockoutPages[0] = 0; // Sentinel
    for( unsigned int i = 0; i < filteredRequests.size(); ++i )
    {
        unsigned int textureId = cascadePageToTextureId( filteredRequests[i] );
        DemandTextureImpl* texture = m_demandLoader->getTexture( textureId );
        unsigned int textureStartPage = texture->getSampler().startPage;
        unsigned int textureEndPage = textureStartPage + texture->getSampler().numPages;
        knockoutPages[ textureStartPage ] = textureEndPage;
    }
    
    // Remove tile requests for textures with cascades and return final request list
    for( unsigned int i = 0; i < numRequests; ++i )
    {
        if( isCascadePage( requests[i] ) )
            continue;

        std::map<unsigned int, unsigned int>::iterator it = knockoutPages.upper_bound( requests[i] );
        if( it != knockoutPages.begin() )
            --it;
        if( requests[i] >= it->first && requests[i] < it->second )
            continue;

        filteredRequests.push_back( requests[i] );
    }
    return filteredRequests;
}

}  // namespace demandLoading
