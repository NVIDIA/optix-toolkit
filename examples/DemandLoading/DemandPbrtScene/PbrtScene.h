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

#pragma once

#include "Dependencies.h"
#include "FrameStopwatch.h"
#include "Scene.h"
#include "SceneProxy.h"
#include "SceneSyncState.h"

#include <OptiXToolkit/DemandLoading/Ticket.h>
#include <OptiXToolkit/Memory/SyncVector.h>

#include <optix.h>

#include <map>
#include <memory>
#include <vector>

namespace demandPbrtScene {

struct Options;
struct Params;

struct SceneGeometry
{
    GeometryInstance instance;
    uint_t           materialId;
    uint_t           instanceIndex;
};

class PbrtScene : public Scene
{
  public:
    PbrtScene( const Options&        options,
               PbrtSceneLoaderPtr    sceneLoader,
               DemandTextureCachePtr demandTextureCache,
               ProxyFactoryPtr       proxyFactory,
               DemandLoaderPtr       demandLoader,
               GeometryLoaderPtr     geometryLoader,
               ProgramGroupsPtr      programGroups,
               MaterialResolverPtr   materialResolver,
               RendererPtr           renderer );
    ~PbrtScene() override = default;

    void initialize( CUstream stream ) override;
    void cleanup() override;

    bool beforeLaunch( CUstream stream, Params& params ) override;
    void afterLaunch( CUstream stream, const Params& params ) override;

    void resolveOneGeometry() override { m_resolveOneGeometry = true; }
    void resolveOneMaterial() override;

    SceneStatistics getStatistics() const override { return m_stats; }

private:
    void                  parseScene();
    void                  realizeInfiniteLights();
    void                  setCamera();
    void                  pushInstance( OptixTraversableHandle handle );
    void                  createTopLevelTraversable( CUstream stream );
    void                  setLaunchParams( CUstream stream, Params& params );
    bool                  resolveProxyGeometry( CUstream stream, uint_t proxyGeomId );
    std::vector<uint_t>   sortRequestedProxyGeometriesByVolume();
    bool                  resolveRequestedProxyGeometries( CUstream stream );

    // Dependencies
    const Options&        m_options;
    PbrtSceneLoaderPtr    m_sceneLoader;
    DemandTextureCachePtr m_demandTextureCache;
    ProxyFactoryPtr       m_proxyFactory;
    DemandLoaderPtr       m_demandLoader;
    GeometryLoaderPtr     m_geometryLoader;
    ProgramGroupsPtr      m_programGroups;
    MaterialResolverPtr   m_materialResolver;
    RendererPtr           m_renderer;

    // Interactive behavior
    bool           m_interactive{};
    bool           m_resolveOneGeometry{};
    FrameStopwatch m_frameTime;

    // Scene related data
    SceneDescriptionPtr             m_scene;
    SceneStatistics                 m_stats{};
    otk::DeviceBuffer               m_tempBuffer;
    otk::DeviceBuffer               m_topLevelAccelBuffer;
    OptixTraversableHandle          m_topLevelTraversable{};
    OptixTraversableHandle          m_proxyInstanceTraversable{};
    demandLoading::Ticket           m_ticket;
    std::map<uint_t, SceneProxyPtr> m_sceneProxies;             // indexed by proxy geometry id
    SceneSyncState                  m_sync;
};

}  // namespace demandPbrtScene
