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

#include "PbrtScene.h"

#include "DemandPbrtSceneKernelCuda.h"
#include "DemandTextureCache.h"
#include "IdRangePrinter.h"
#include "ImageSourceFactory.h"
#include "MaterialResolver.h"
#include "Options.h"
#include "Params.h"
#include "ProgramGroups.h"
#include "Renderer.h"
#include "SceneAdapters.h"
#include "SceneProxy.h"
#include "Stopwatch.h"

#include <OptiXToolkit/DemandGeometry/GeometryLoader.h>
#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>
#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneLoader.h>

#include <optix_stubs.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace demandPbrtScene {

PbrtScene::PbrtScene( const Options&        options,
                      PbrtSceneLoaderPtr    sceneLoader,
                      DemandTextureCachePtr demandTextureCache,
                      ProxyFactoryPtr       proxyFactory,
                      DemandLoaderPtr       demandLoader,
                      GeometryLoaderPtr     geometryLoader,
                      ProgramGroupsPtr      programGroups,
                      MaterialResolverPtr   materialResolver,
                      RendererPtr           renderer )
    : m_options( options )
    , m_sceneLoader( std::move( sceneLoader ) )
    , m_demandTextureCache( std::move( demandTextureCache ) )
    , m_proxyFactory( std::move( proxyFactory ) )
    , m_demandLoader( std::move( demandLoader ) )
    , m_geometryLoader( std::move( geometryLoader ) )
    , m_programGroups( std::move( programGroups ) )
    , m_materialResolver( std::move( materialResolver ) )
    , m_renderer( std::move( renderer ) )
    , m_interactive( m_options.outFile.empty() )
    , m_frameTime( m_interactive )
{
}

inline OptixAabb toOptixAabb( const ::pbrt::Bounds3f& bounds )
{
    return OptixAabb{ bounds.pMin.x, bounds.pMin.y, bounds.pMin.z, bounds.pMax.x, bounds.pMax.y, bounds.pMax.z };
}

void PbrtScene::realizeInfiniteLights()
{
    m_sync.infiniteLights.resize( m_scene->infiniteLights.size() );
    bool first{ true };
    for( size_t i = 0; i < m_scene->infiniteLights.size(); ++i )
    {
        const otk::pbrt::InfiniteLightDefinition& src = m_scene->infiniteLights[i];

        InfiniteLight& dest = m_sync.infiniteLights[i];
        dest.color          = make_float3( src.color.x, src.color.y, src.color.z );
        dest.scale          = make_float3( src.scale.x, src.scale.y, src.scale.z );
        // only one skybox texture supported
        if( !src.environmentMapName.empty() && first )
        {
            dest.skyboxTextureId = m_demandTextureCache->createSkyboxTextureFromFile( src.environmentMapName );
            first                = false;
        }
    }
}

void PbrtScene::setCamera()
{
    auto fromPoint3f  = []( const pbrt::Point3f& pt ) { return make_float3( pt.x, pt.y, pt.z ); };
    auto fromVector3f = []( const pbrt::Vector3f& vec ) { return make_float3( vec.x, vec.y, vec.z ); };

    PerspectiveCamera camera;
    LookAtParams      lookAt;
    // TODO: handle lookAt and camera keywords from file properly with current transformation matrix
    const otk::pbrt::LookAtDefinition& sceneLookAt = m_scene->lookAt;
    if( sceneLookAt.lookAt == ::pbrt::Point3f() && sceneLookAt.eye == ::pbrt::Point3f() && sceneLookAt.up == ::pbrt::Vector3f() )
    {
        lookAt.lookAt = make_float3( 0.0f, 0.0f, -3.0f );
        lookAt.eye    = make_float3( 0.0f, 0.0f, 0.0f );
        lookAt.up     = make_float3( 0.0f, 1.0f, 0.0f );
    }
    else
    {
        lookAt.lookAt = fromPoint3f( sceneLookAt.lookAt );
        lookAt.eye    = fromPoint3f( sceneLookAt.eye );
        lookAt.up     = fromVector3f( sceneLookAt.up );
    }
    m_renderer->setLookAt( lookAt );
    const otk::pbrt::PerspectiveCameraDefinition& sceneCamera = m_scene->camera;

    camera.fovY          = sceneCamera.fov;
    camera.focalDistance = sceneCamera.focalDistance;
    camera.lensRadius    = sceneCamera.lensRadius;
    camera.aspectRatio   = static_cast<float>( m_options.width ) / static_cast<float>( m_options.height );
    toFloat4Transform( camera.cameraToWorld.m, sceneCamera.cameraToWorld.GetMatrix() );
    toFloat4Transform( camera.worldToCamera.m, sceneCamera.cameraToWorld.GetInverseMatrix() );
    toFloat4Transform( camera.cameraToScreen.m, sceneCamera.cameraToScreen.GetMatrix() );
    m_renderer->setCamera( camera );
}

static void identity( float ( &result )[12] )
{
    const float matrix[12]{
        1.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 1.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 1.0f, 0.0f   //
    };
    std::copy( std::begin( matrix ), std::end( matrix ), std::begin( result ) );
}

void PbrtScene::pushInstance( OptixTraversableHandle handle )
{
    OptixInstance instance;
    identity( instance.transform );
    instance.instanceId        = 0;
    instance.sbtOffset         = +HitGroupIndex::PROXY_GEOMETRY;
    instance.visibilityMask    = 255U;
    instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
    instance.traversableHandle = handle;
    m_sync.topLevelInstances.push_back( instance );
}

void PbrtScene::createTopLevelTraversable( CUstream stream )
{
    m_sync.topLevelInstances.copyToDeviceAsync( stream );
    if( m_options.sync )
    {
        OTK_CUDA_SYNC_CHECK();
    }

    const uint_t    NUM_BUILD_INPUTS = 1;
    OptixBuildInput inputs[NUM_BUILD_INPUTS]{};
    otk::BuildInputBuilder( inputs ).instanceArray( m_sync.topLevelInstances, m_sync.topLevelInstances.size() );

    OptixDeviceContext           context = m_renderer->getDeviceContext();
    const OptixAccelBuildOptions options = { OPTIX_BUILD_FLAG_NONE,        // buildFlags
                                             OPTIX_BUILD_OPERATION_BUILD,  // operation
                                             OptixMotionOptions{ /*numKeys=*/0, /*flags=*/0, /*timeBegin=*/0.f, /*timeEnd=*/0.f } };
    OptixAccelBufferSizes        sizes{};
    OTK_ERROR_CHECK( optixAccelComputeMemoryUsage( context, &options, inputs, NUM_BUILD_INPUTS, &sizes ) );
    if( m_options.sync )
    {
        OTK_CUDA_SYNC_CHECK();
    }

    m_tempBuffer.resize( sizes.tempSizeInBytes );
    m_topLevelAccelBuffer.resize( sizes.outputSizeInBytes );
    OTK_ERROR_CHECK( optixAccelBuild( context, stream, &options, inputs, NUM_BUILD_INPUTS, m_tempBuffer,
                                      m_tempBuffer.capacity(), m_topLevelAccelBuffer, m_topLevelAccelBuffer.capacity(),
                                      &m_topLevelTraversable, nullptr, 0 ) );
    if( m_options.sync )
    {
        OTK_CUDA_SYNC_CHECK();
    }
}

static PbrtFileStatistics getPbrtStatistics( const std::string& filePath, SceneDescriptionPtr scene, double parseTime )
{
    const auto         asUInt{ []( const size_t value ) { return static_cast<unsigned int>( value ); } };
    PbrtFileStatistics stats{};
    stats.fileName           = std::filesystem::path( filePath ).filename().string();
    stats.numFreeShapes      = asUInt( scene->freeShapes.size() );
    stats.numObjects         = asUInt( scene->objects.size() );
    stats.numObjectShapes    = asUInt( scene->objectShapes.size() );
    stats.numObjectInstances = asUInt( scene->objectInstances.size() );
    stats.parseTime          = parseTime;
    return stats;
}

void PbrtScene::resolveOneMaterial()
{
    m_materialResolver->resolveOneMaterial();
}

void PbrtScene::parseScene()
{
    std::cout << "Reading " << m_options.sceneFile << "...\n";
    Stopwatch timer;
    m_scene = m_sceneLoader->parseFile( m_options.sceneFile );
    const double parseTime{ timer.elapsed() };
    if( m_scene->errors > 0 )
    {
        throw std::runtime_error( "Couldn't load scene " + m_options.sceneFile );
    }
    m_stats.pbrtFile = getPbrtStatistics( m_options.sceneFile, m_scene, parseTime );
    std::cout << "\nParsed scene " << m_options.sceneFile << " in " << m_stats.pbrtFile.parseTime
              << " secs, loaded: " << m_stats.pbrtFile.numFreeShapes << " free shapes, " << m_stats.pbrtFile.numObjects
              << " objects, " << m_stats.pbrtFile.numObjectInstances << " instances.\n";

    SceneProxyPtr proxy{ m_proxyFactory->scene( m_geometryLoader, m_scene ) };
    m_sceneProxies[proxy->getPageId()] = proxy;
}

void PbrtScene::initialize( CUstream stream )
{
    parseScene();
    setCamera();

    m_programGroups->initialize();

    m_geometryLoader->setSbtIndex( +HitGroupIndex::PROXY_GEOMETRY );
    m_geometryLoader->copyToDeviceAsync( stream );
    m_proxyInstanceTraversable = m_geometryLoader->createTraversable( m_renderer->getDeviceContext(), stream );
    if( m_options.sync )
    {
        OTK_CUDA_SYNC_CHECK();
    }

    m_sync.topLevelInstances.clear();
    pushInstance( m_proxyInstanceTraversable );
    createTopLevelTraversable( stream );
}

void PbrtScene::cleanup()
{
    m_programGroups->cleanup();
}

void PbrtScene::setLaunchParams( CUstream stream, Params& params )
{
    static const float3 proxyFaceColors[6] = {
        { 1.0f, 1.0f, 0.0f },  // +x
        { 1.0f, 0.0f, 0.0f },  // -x
        { 0.0f, 1.0f, 1.0f },  // +y
        { 0.0f, 1.0f, 0.0f },  // -y
        { 1.0f, 0.0f, 1.0f },  // +z
        { 0.0f, 0.0f, 1.0f },  // -z
    };
    std::copy( std::begin( proxyFaceColors ), std::end( proxyFaceColors ), std::begin( params.proxyFaceColors ) );

    params.ambientColor           = make_float3( 0.4f, 0.4f, 0.4f );
    params.sceneEpsilon           = 0.005f;
    params.usePinholeCamera       = m_options.usePinholeCamera;
    params.traversable            = m_topLevelTraversable;
    params.demandGeomContext      = m_geometryLoader->getContext();
    const float3 yellow           = make_float3( 1.0f, 1.0f, 0.0 );
    params.demandMaterialColor    = yellow;
    params.numPartialMaterials    = static_cast<uint_t>( m_sync.partialMaterials.size() );
    params.partialMaterials       = m_sync.partialMaterials.typedDevicePtr();
    params.numRealizedMaterials   = static_cast<uint_t>( m_sync.realizedMaterials.size() );
    params.realizedMaterials      = m_sync.realizedMaterials.typedDevicePtr();
    params.numInstanceMaterialIds = static_cast<uint_t>( m_sync.instanceMaterialIds.size() );
    params.instanceMaterialIds    = m_sync.instanceMaterialIds.typedDevicePtr();
    params.numInstanceNormals     = static_cast<uint_t>( m_sync.realizedNormals.size() );
    params.instanceNormals        = m_sync.realizedNormals.typedDevicePtr();
    params.numInstanceUVs         = static_cast<uint_t>( m_sync.realizedUVs.size() );
    params.instanceUVs            = m_sync.realizedUVs.typedDevicePtr();
    params.numPartialUVs          = static_cast<uint_t>( m_sync.partialUVs.size() );
    params.partialUVs             = m_sync.partialUVs.typedDevicePtr();
    // Copy lights from the scene description; only need to do this once or if lights change interactively.
    if( params.numDirectionalLights == 0 && !m_scene->distantLights.empty() )
    {
        m_sync.directionalLights.resize( m_scene->distantLights.size() );
        std::transform( m_scene->distantLights.begin(), m_scene->distantLights.end(), m_sync.directionalLights.begin(),
                        []( const otk::pbrt::DistantLightDefinition& distant ) {
                            const pbrt::Vector3f direction( Normalize( distant.lightToWorld( distant.direction ) ) );
                            return DirectionalLight{ make_float3( direction.x, direction.y, direction.z ),
                                                     make_float3( distant.color.x * distant.scale.x,
                                                                  distant.color.y * distant.scale.y,
                                                                  distant.color.z * distant.scale.z ) };
                        } );
        m_sync.directionalLights.copyToDevice();
        params.numDirectionalLights = static_cast<uint_t>( m_sync.directionalLights.size() );
        params.directionalLights    = m_sync.directionalLights.typedDevicePtr();
    }
    if( params.numInfiniteLights == 0 && !m_scene->infiniteLights.empty() )
    {
        realizeInfiniteLights();
        m_sync.infiniteLights.copyToDevice();
        params.numInfiniteLights = static_cast<uint_t>( m_sync.infiniteLights.size() );
        params.infiniteLights    = m_sync.infiniteLights.typedDevicePtr();
    }
    params.minAlphaTextureId   = m_sync.minAlphaTextureId;
    params.maxAlphaTextureId   = m_sync.maxAlphaTextureId;
    params.minDiffuseTextureId = m_sync.minDiffuseTextureId;
    params.maxDiffuseTextureId = m_sync.maxDiffuseTextureId;

    m_demandLoader->launchPrepare( stream, params.demandContext );
}

bool PbrtScene::resolveProxyGeometry( CUstream stream, uint_t proxyGeomId )
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
        geom.instance = removedProxy->createGeometry( m_renderer->getDeviceContext(), stream );
        updateNeeded  = m_materialResolver->resolveMaterialForGeometry( proxyGeomId, geom, m_sync );
        ++m_stats.numGeometriesRealized;
    }
    ++m_stats.numProxyGeometriesResolved;

    return updateNeeded;
}

inline float volume( const OptixAabb& bounds )
{
    return std::fabs( bounds.maxX - bounds.minX ) * std::fabs( bounds.maxY - bounds.minY )
           * std::fabs( bounds.maxZ - bounds.minZ );
}


std::vector<uint_t> PbrtScene::sortRequestedProxyGeometriesByVolume()
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

bool PbrtScene::resolveRequestedProxyGeometries( CUstream stream )
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
        if( m_frameTime.expired() && realizedCount > MIN_REALIZED )
        {
            break;
        }
        ++realizedCount;

        if( resolveProxyGeometry( stream, id ) )
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
            m_sync.realizedNormals.copyToDeviceAsync( stream );
            m_sync.realizedUVs.copyToDeviceAsync( stream );
            m_sync.instanceMaterialIds.copyToDeviceAsync( stream );
        }

        m_geometryLoader->copyToDeviceAsync( stream );
        OptixDeviceContext context{ m_renderer->getDeviceContext() };
        m_proxyInstanceTraversable                    = m_geometryLoader->createTraversable( context, stream );
        m_sync.topLevelInstances[0].traversableHandle = m_proxyInstanceTraversable;
    }

    return realized;
}

bool PbrtScene::beforeLaunch( CUstream stream, Params& params )
{
    m_ticket.wait();
    m_frameTime.start();

    const MaterialResolution realizedMaterial = m_materialResolver->resolveRequestedProxyMaterials( stream, m_frameTime, m_sync );
    const bool realizedGeometry = resolveRequestedProxyGeometries( stream );

    if( realizedGeometry || realizedMaterial != MaterialResolution::NONE )
    {
        createTopLevelTraversable( stream );
    }

    setLaunchParams( stream, params );

    // Render needed if we changed accels or we still have unresolved proxies.
    return realizedGeometry || realizedMaterial != MaterialResolution::NONE || !m_sceneProxies.empty()
           || m_options.oneShotGeometry || m_options.oneShotMaterial || m_ticket.numTasksTotal() > 0;
}

void PbrtScene::afterLaunch( CUstream stream, const Params& params )
{
    m_ticket = m_demandLoader->processRequests( stream, params.demandContext );
}

ScenePtr createScene( const Options&        options,
                      PbrtSceneLoaderPtr    sceneLoader,
                      DemandTextureCachePtr demandTextureCache,
                      ProxyFactoryPtr       proxyFactory,
                      DemandLoaderPtr       demandLoader,
                      GeometryLoaderPtr     geometryLoader,
                      ProgramGroupsPtr      programGroups,
                      MaterialResolverPtr   materialResolver,
                      RendererPtr           renderer )
{
    return std::make_shared<PbrtScene>( options, std::move( sceneLoader ), std::move( demandTextureCache ),
                                        std::move( proxyFactory ), std::move( demandLoader ), std::move( geometryLoader ),
                                        std::move( programGroups ), std::move( materialResolver ), std::move( renderer ) );
}

}  // namespace demandPbrtScene
