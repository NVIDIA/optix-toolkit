// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/Scene.h"

#include "DemandPbrtScene/DemandTextureCache.h"
#include "DemandPbrtScene/FrameStopwatch.h"
#include "DemandPbrtScene/GeometryResolver.h"
#include "DemandPbrtScene/MaterialResolver.h"
#include "DemandPbrtScene/Options.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/ProgramGroups.h"
#include "DemandPbrtScene/Renderer.h"
#include "DemandPbrtScene/SceneAdapters.h"
#include "DemandPbrtScene/SceneSyncState.h"
#include "DemandPbrtScene/Stopwatch.h"

#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneLoader.h>
#include <OptiXToolkit/Memory/DeviceBuffer.h>

#include <optix_stubs.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>

namespace demandPbrtScene {

namespace {

class PbrtScene : public Scene
{
  public:
    PbrtScene( const Options&        options,
               PbrtSceneLoaderPtr    sceneLoader,
               DemandTextureCachePtr demandTextureCache,
               DemandLoaderPtr       demandLoader,
               MaterialResolverPtr   materialResolver,
               GeometryResolverPtr   geometryResolver,
               RendererPtr           renderer );
    ~PbrtScene() override = default;

    void initialize( CUstream stream ) override;

    bool beforeLaunch( CUstream stream, Params& params ) override;
    void afterLaunch( CUstream stream, const Params& params ) override;

    void resolveOneGeometry() override;
    void resolveOneMaterial() override;

    SceneStatistics getStatistics() const override { return m_stats; }

  private:
    void parseScene();
    void realizeInfiniteLights();
    void setCamera();
    void createTopLevelTraversable( CUstream stream );
    void setLaunchParams( CUstream stream, Params& params );

    // Dependencies
    const Options&        m_options;
    PbrtSceneLoaderPtr    m_sceneLoader;
    DemandTextureCachePtr m_demandTextureCache;
    DemandLoaderPtr       m_demandLoader;
    MaterialResolverPtr   m_materialResolver;
    GeometryResolverPtr   m_geometryResolver;
    RendererPtr           m_renderer;

    // Interactive behavior
    bool           m_interactive{};
    FrameStopwatch m_frameTime;

    // Scene related data
    SceneDescriptionPtr    m_scene;
    SceneStatistics        m_stats{};
    otk::DeviceBuffer      m_tempBuffer;
    otk::DeviceBuffer      m_topLevelAccelBuffer;
    OptixTraversableHandle m_topLevelTraversable{};
    demandLoading::Ticket  m_ticket;
    SceneSyncState         m_sync;
};

PbrtScene::PbrtScene( const Options&        options,
                      PbrtSceneLoaderPtr    sceneLoader,
                      DemandTextureCachePtr demandTextureCache,
                      DemandLoaderPtr       demandLoader,
                      MaterialResolverPtr   materialResolver,
                      GeometryResolverPtr   geometryResolver,
                      RendererPtr           renderer )
    : m_options( options )
    , m_sceneLoader( std::move( sceneLoader ) )
    , m_demandTextureCache( std::move( demandTextureCache ) )
    , m_demandLoader( std::move( demandLoader ) )
    , m_materialResolver( std::move( materialResolver ) )
    , m_geometryResolver( std::move( geometryResolver ) )
    , m_renderer( std::move( renderer ) )
    , m_interactive( m_options.outFile.empty() )
    , m_frameTime( m_interactive )
{
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

static SceneStatistics getSceneStatistics( const std::string& filePath, SceneDescriptionPtr scene, double parseTime )
{
    const auto      asUInt{ []( const size_t value ) { return static_cast<unsigned int>( value ); } };
    SceneStatistics stats{};
    stats.fileName           = std::filesystem::path( filePath ).filename().string();
    stats.numFreeShapes      = asUInt( scene->freeShapes.size() );
    stats.numObjects         = asUInt( scene->objects.size() );
    stats.numObjectShapes    = asUInt( scene->objectShapes.size() );
    stats.numObjectInstances = asUInt( scene->objectInstances.size() );
    stats.parseTime          = parseTime;
    return stats;
}

void PbrtScene::resolveOneGeometry()
{
    m_geometryResolver->resolveOneGeometry();
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
    m_stats = getSceneStatistics( m_options.sceneFile, m_scene, parseTime );
    std::cout << "\nParsed scene " << m_options.sceneFile << " in " << m_stats.parseTime << " secs, loaded: " << m_stats.numFreeShapes
              << " free shapes, " << m_stats.numObjects << " objects, " << m_stats.numObjectInstances << " instances.\n";
}

void PbrtScene::initialize( CUstream stream )
{
    parseScene();
    setCamera();
    m_geometryResolver->initialize( stream, m_renderer->getDeviceContext(), m_scene, m_sync );
    createTopLevelTraversable( stream );
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
    params.demandGeomContext      = m_geometryResolver->getContext();
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

bool PbrtScene::beforeLaunch( CUstream stream, Params& params )
{
    m_ticket.wait();
    m_frameTime.start();

    const OptixDeviceContext context{ m_renderer->getDeviceContext() };
    const MaterialResolution realizedMaterial{ m_materialResolver->resolveRequestedProxyMaterials( stream, m_frameTime, m_sync ) };
    const bool realizedGeometry{ m_geometryResolver->resolveRequestedProxyGeometries( stream, context, m_frameTime, m_sync ) };

    if( realizedGeometry || realizedMaterial != MaterialResolution::NONE )
    {
        createTopLevelTraversable( stream );
    }

    setLaunchParams( stream, params );

    // Render needed if we changed accels or we still have remaining work to do.
    return realizedGeometry || realizedMaterial != MaterialResolution::NONE  // some proxy was resolved
           || m_options.oneShotGeometry || m_options.oneShotMaterial         // something is in one shot mode
           || m_ticket.numTasksTotal() > 0;                                  // some demand loaded data was requested
}

void PbrtScene::afterLaunch( CUstream stream, const Params& params )
{
    m_ticket = m_demandLoader->processRequests( stream, params.demandContext );
}

}  // namespace

ScenePtr createScene( const Options&        options,
                      PbrtSceneLoaderPtr    sceneLoader,
                      DemandTextureCachePtr demandTextureCache,
                      DemandLoaderPtr       demandLoader,
                      MaterialResolverPtr   materialResolver,
                      GeometryResolverPtr   geometryResolver,
                      RendererPtr           renderer )
{
    return std::make_shared<PbrtScene>( options,                          //
                                        std::move( sceneLoader ),         //
                                        std::move( demandTextureCache ),  //
                                        std::move( demandLoader ),        //
                                        std::move( materialResolver ),    //
                                        std::move( geometryResolver ),    //
                                        std::move( renderer ) );
}

}  // namespace demandPbrtScene
