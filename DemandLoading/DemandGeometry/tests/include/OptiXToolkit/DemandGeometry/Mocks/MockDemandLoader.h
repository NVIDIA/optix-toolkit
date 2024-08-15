// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
                   int                                            baseTextureId,
                   unsigned int                                   numChannelTextures ) );
    MOCK_METHOD( unsigned int, createResource, ( unsigned int numPages, demandLoading::ResourceCallback callback, void* callbackContext ) );
    MOCK_METHOD( void, invalidatePage, ( unsigned int pageId ) );
    MOCK_METHOD( void, loadTextureTiles, ( CUstream stream, unsigned int textureId, bool reloadIfResident ) );
    MOCK_METHOD( void, unloadTextureTiles, ( unsigned int textureId ) );
    MOCK_METHOD( void, setPageTableEntry, ( unsigned int pageId, bool evictable, unsigned long long pageTableEntry ) );
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
