#include "GeometryResolver.h"

#include <OptiXToolkit/DemandGeometry/GeometryLoader.h>
#include <OptiXToolkit/Memory/SyncVector.h>

#include "DemandTextureCache.h"
#include "FrameStopwatch.h"
#include "IdRangePrinter.h"
#include "MaterialResolver.h"
#include "Options.h"
#include "ProgramGroups.h"
#include "Renderer.h"
#include "SceneGeometry.h"
#include "SceneProxy.h"
#include "SceneSyncState.h"

#include <optix.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

namespace demandPbrtScene {

namespace {

class PbrtGeometryResolver : public GeometryResolver
{
  public:
    PbrtGeometryResolver( const Options&          options,
                          ProgramGroupsPtr&&      programGroups,
                          GeometryLoaderPtr&&     geometryLoader,
                          ProxyFactoryPtr&&       proxyFactory,
                          DemandTextureCachePtr&& demandTextureCache,
                          MaterialResolverPtr&&   materialResolver )
        : m_options( options )
        , m_programGroups( std::move( programGroups ) )
        , m_geometryLoader( std::move( geometryLoader ) )
        , m_proxyFactory( std::move( proxyFactory ) )
        , m_demandTextureCache( std::move( demandTextureCache ) )
        , m_materialResolver( std::move( materialResolver ) )
    {
    }
    ~PbrtGeometryResolver() override = default;

    void initialize( CUstream stream, OptixDeviceContext context, const SceneDescriptionPtr& scene, SceneSyncState& sync ) override;
    demandGeometry::Context getContext() const override { return m_geometryLoader->getContext(); }
    void resolveOneGeometry() override { m_resolveOneGeometry = true; }
    bool resolveRequestedProxyGeometries( CUstream stream, OptixDeviceContext context, const FrameStopwatch& frameTime, SceneSyncState& sync ) override;

    GeometryResolverStatistics getStatistics() const override { return m_stats; }

  private:
    void                pushInstance( SceneSyncState& sync, OptixTraversableHandle handle );
    std::vector<uint_t> sortRequestedProxyGeometriesByVolume();
    bool resolveProxyGeometry( CUstream stream, OptixDeviceContext context, uint_t proxyGeomId, SceneSyncState& m_sync );

    // Dependencies
    const Options&        m_options;
    ProgramGroupsPtr      m_programGroups;
    GeometryLoaderPtr     m_geometryLoader;
    ProxyFactoryPtr       m_proxyFactory;
    DemandTextureCachePtr m_demandTextureCache;
    MaterialResolverPtr   m_materialResolver;

    // Scene data
    OptixTraversableHandle          m_proxyInstanceTraversable{};
    std::map<uint_t, SceneProxyPtr> m_sceneProxies;  // indexed by proxy geometry id
    bool                            m_resolveOneGeometry{};
    GeometryResolverStatistics      m_stats{};
};

static void identity( float ( &result )[12] )
{
    const float matrix[12]{
        1.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 1.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 1.0f, 0.0f   //
    };
    std::copy( std::begin( matrix ), std::end( matrix ), std::begin( result ) );
}

void PbrtGeometryResolver::pushInstance( SceneSyncState& sync, OptixTraversableHandle handle )
{
    OptixInstance instance;
    identity( instance.transform );
    instance.instanceId        = 0;
    instance.sbtOffset         = +HitGroupIndex::PROXY_GEOMETRY;
    instance.visibilityMask    = 255U;
    instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
    instance.traversableHandle = handle;
    sync.topLevelInstances.push_back( instance );
}

void PbrtGeometryResolver::initialize( CUstream stream, OptixDeviceContext context, const SceneDescriptionPtr& scene, SceneSyncState& sync )
{
    SceneProxyPtr proxy{ m_proxyFactory->scene( scene) };
    m_sceneProxies[proxy->getPageId()] = proxy;
    m_geometryLoader->setSbtIndex( +HitGroupIndex::PROXY_GEOMETRY );
    m_geometryLoader->copyToDeviceAsync( stream );
    m_proxyInstanceTraversable         = m_geometryLoader->createTraversable( context, stream );
    if( m_options.sync )
    {
        OTK_CUDA_SYNC_CHECK();
    }
    sync.topLevelInstances.clear();
    pushInstance( sync, m_proxyInstanceTraversable );
}

float volume( const OptixAabb& bounds )
{
    return std::fabs( bounds.maxX - bounds.minX ) * std::fabs( bounds.maxY - bounds.minY )
           * std::fabs( bounds.maxZ - bounds.minZ );
}

std::vector<uint_t> PbrtGeometryResolver::sortRequestedProxyGeometriesByVolume()
{
    std::vector<uint_t> ids{ m_geometryLoader->requestedProxyIds() };
    if( m_options.sortProxies )
    {
        std::sort( ids.begin(), ids.end(), [this]( const uint_t lhs, const uint_t rhs ) {
            const float lhsVolume = volume( m_sceneProxies[lhs]->getBounds() );
            const float rhsVolume = volume( m_sceneProxies[rhs]->getBounds() );
            return lhsVolume > rhsVolume;
        } );
    }
    return ids;
}

bool PbrtGeometryResolver::resolveProxyGeometry( CUstream stream, OptixDeviceContext context, uint_t proxyGeomId, SceneSyncState& m_sync )
{
    bool updateNeeded{};

    m_geometryLoader->remove( proxyGeomId );
    auto it = m_sceneProxies.find( proxyGeomId );
    if( it == m_sceneProxies.end() )
    {
        throw std::runtime_error( "Proxy geometry " + std::to_string( proxyGeomId ) + " not found" );
    }

    // Remove proxy from scene proxies map.
    SceneProxyPtr removedProxy = it->second;
    m_sceneProxies.erase( proxyGeomId );

    // Add replacement for the proxy to the scene
    if( removedProxy->isDecomposable() )
    {
        static std::vector<uint_t> subProxies;
        subProxies.clear();

        // get sub-proxies and add to scene
        for( SceneProxyPtr proxy : removedProxy->decompose( m_geometryLoader, m_proxyFactory ) )
        {
            const uint_t id = proxy->getPageId();
            subProxies.push_back( id );
            m_sceneProxies[id] = proxy;
        }
        if( m_options.verboseProxyGeometryResolution )
        {
            std::cout << "Resolved proxy geometry id " << proxyGeomId << " to "
                      << ( subProxies.size() > 1 ? "ids " : "id " ) << IdRange{ subProxies } << '\n';
        }
    }
    else
    {
        // add instance to TLAS instances
        SceneGeometry geom;
        geom.instance = removedProxy->createGeometry( context, stream );
        updateNeeded  = m_materialResolver->resolveMaterialForGeometry( proxyGeomId, geom, m_sync );
        ++m_stats.numGeometriesRealized;
    }
    ++m_stats.numProxyGeometriesResolved;

    return updateNeeded;
}

bool PbrtGeometryResolver::resolveRequestedProxyGeometries( CUstream              stream,
                                                            OptixDeviceContext    context,
                                                            const FrameStopwatch& frameTime,
                                                            SceneSyncState&       sync )
{
    if( m_options.oneShotGeometry && !m_resolveOneGeometry )
    {
        return false;
    }

    const unsigned int MIN_REALIZED{ 512 };
    unsigned int       realizedCount{};
    bool               realized{};
    bool               updateNeeded{};
    for( uint_t id : sortRequestedProxyGeometriesByVolume() )
    {
        if( frameTime.expired() && realizedCount > MIN_REALIZED )
        {
            break;
        }
        ++realizedCount;

        if( resolveProxyGeometry( stream, context, id, sync ) )
        {
            updateNeeded = true;
        }
        realized = true;

        if( m_resolveOneGeometry )
        {
            m_resolveOneGeometry = false;
            break;
        }
    }
    m_geometryLoader->clearRequestedProxyIds();

    if( realized )
    {
        if( updateNeeded )
        {
            // we reused a realized material while resolving a proxy geometry
            sync.realizedNormals.copyToDeviceAsync( stream );
            sync.realizedUVs.copyToDeviceAsync( stream );
            sync.instanceMaterialIds.copyToDeviceAsync( stream );
        }

        m_geometryLoader->copyToDeviceAsync( stream );
        m_proxyInstanceTraversable                  = m_geometryLoader->createTraversable( context, stream );
        sync.topLevelInstances[0].traversableHandle = m_proxyInstanceTraversable;
    }

    return realized;
}

}  // namespace

GeometryResolverPtr createGeometryResolver( const Options&        options,
                                            ProgramGroupsPtr      programGroups,
                                            GeometryLoaderPtr     geometryLoader,
                                            ProxyFactoryPtr       proxyFactory,
                                            DemandTextureCachePtr demandTextureCache,
                                            MaterialResolverPtr   materialResolver )
{
    return std::make_shared<PbrtGeometryResolver>( options,                          //
                                                   std::move( programGroups ),       //
                                                   std::move( geometryLoader ),      //
                                                   std::move( proxyFactory ),        //
                                                   std::move( demandTextureCache ),  //
                                                   std::move( materialResolver ) );
}

}  // namespace demandPbrtScene
