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

#include <stdio.h>
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
