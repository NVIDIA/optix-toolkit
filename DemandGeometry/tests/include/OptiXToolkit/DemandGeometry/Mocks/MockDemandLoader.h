//
//  Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#pragma once

#include <OptiXToolkit/DemandLoading/DemandLoader.h>

#include <gmock/gmock.h>

namespace otk {
namespace testing {

class MockDemandLoader : public demandLoading::DemandLoader
{
  public:
    MOCK_METHOD( const demandLoading::DemandTexture&,
                 createTexture,
                 ( std::shared_ptr<imageSource::ImageSource> image, const demandLoading::TextureDescriptor& textureDesc ) );
    MOCK_METHOD( const demandLoading::DemandTexture&,
                 createUdimTexture,
                 ( std::vector<std::shared_ptr<imageSource::ImageSource>> & imageSources,
                   std::vector<demandLoading::TextureDescriptor>& textureDescs,
                   unsigned int                                   udim,
                   unsigned int                                   vdim,
                   int                                            baseTextureId ) );
    MOCK_METHOD( unsigned int, createResource, ( unsigned int numPages, demandLoading::ResourceCallback callback, void* callbackContext ) );
    MOCK_METHOD( void, unloadResource, ( unsigned int pageId ) );
    MOCK_METHOD( void, unloadTextureTiles, ( unsigned int textureId ) );
    MOCK_METHOD( void,
                 replaceTexture,
                 ( CUstream stream, unsigned int textureId, std::shared_ptr<imageSource::ImageSource> image, const demandLoading::TextureDescriptor& textureDesc, bool migrateTiles ) );
    MOCK_METHOD( bool, launchPrepare, ( unsigned int deviceIndex, CUstream stream, demandLoading::DeviceContext& context ) );
    MOCK_METHOD( demandLoading::Ticket,
                 processRequests,
                 ( unsigned int deviceIndex, CUstream stream, const demandLoading::DeviceContext& deviceContext ) );
    MOCK_METHOD( void, abort, () );
    MOCK_METHOD( demandLoading::Statistics, getStatistics, (), ( const ) );
    MOCK_METHOD( std::vector<unsigned int>, getDevices, (), ( const ) );
    MOCK_METHOD( const demandLoading::Options&, getOptions, () );
    MOCK_METHOD( void, enableEviction, ( bool evictionActive ) );
    MOCK_METHOD( void, setMaxTextureMemory, ( size_t maxMem ) );
    MOCK_METHOD( const demandLoading::Options&, getOptions, (), ( const ) );
    MOCK_METHOD( void, initTexture, (CUstream, unsigned int), ( override ) );
    MOCK_METHOD( void, initUdimTexture, (CUstream, unsigned int), ( override ) );
    MOCK_METHOD( unsigned int, getTextureTilePageId, (unsigned int, unsigned int, unsigned int, unsigned int), ( override ) );
    MOCK_METHOD( unsigned int, getMipTailFirstLevel, (unsigned int), ( override ) );
    MOCK_METHOD( void, loadTextureTile, (CUstream, unsigned int, unsigned int, unsigned int, unsigned int), ( override ) );
    MOCK_METHOD( bool, pageResident, (unsigned int), ( override ) );
    MOCK_METHOD( bool, launchPrepare, (CUstream, demandLoading::DeviceContext&), ( override ) );
    MOCK_METHOD( demandLoading::Ticket, processRequests, (CUstream, const demandLoading::DeviceContext&), ( override ) );
    MOCK_METHOD( CUcontext, getCudaContext, (), ( override ) );
};

}  // namespace testing
}  // namespace otk
