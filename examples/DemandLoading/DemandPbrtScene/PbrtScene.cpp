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
#include "Options.h"
#include "Params.h"
#include "Renderer.h"
#include "SceneAdapters.h"
#include "SceneProxy.h"

#include <OptiXToolkit/DemandGeometry/GeometryLoader.h>
#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneLoader.h>

#include <optix_stubs.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif

namespace demandPbrtScene {

PbrtScene::PbrtScene( const Options&        options,
                      PbrtSceneLoaderPtr    pbrt,
                      DemandTextureCachePtr demandTextureCache,
                      ProxyFactoryPtr       proxyFactory,
                      DemandLoaderPtr       demandLoader,
                      GeometryLoaderPtr     geometryLoader,
                      MaterialLoaderPtr     materialLoader,
                      RendererPtr           renderer )
    : m_options( options )
    , m_sceneLoader( std::move( pbrt ) )
    , m_demandTextureCache( std::move( demandTextureCache ) )
    , m_proxyFactory( std::move( proxyFactory ) )
    , m_demandLoader( std::move( demandLoader ) )
    , m_geometryLoader( std::move( geometryLoader ) )
    , m_materialLoader( std::move( materialLoader ) )
    , m_renderer( std::move( renderer ) )
    , m_interactive( m_options.outFile.empty() )
{
}

inline OptixAabb toOptixAabb( const ::pbrt::Bounds3f& bounds )
{
    return OptixAabb{ bounds.pMin.x, bounds.pMin.y, bounds.pMin.z, bounds.pMax.x, bounds.pMax.y, bounds.pMax.z };
}

static OptixModuleCompileOptions getCompileOptions()
{
    OptixModuleCompileOptions compileOptions{};
    compileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NDEBUG
    bool debugInfo{ false };
#else
    bool debugInfo{ true };
#endif
    compileOptions.optLevel   = debugInfo ? OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 : OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    compileOptions.debugLevel = debugInfo ? OPTIX_COMPILE_DEBUG_LEVEL_FULL : OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    return compileOptions;
}

void PbrtScene::realizeInfiniteLights()
{
    m_infiniteLights.resize( m_scene->infiniteLights.size() );
    bool first{ true };
    for( size_t i = 0; i < m_scene->infiniteLights.size(); ++i )
    {
        const otk::pbrt::InfiniteLightDefinition& src  = m_scene->infiniteLights[i];

        InfiniteLight& dest = m_infiniteLights[i];
        dest.color          = make_float3( src.color.x, src.color.y, src.color.z );
        dest.scale          = make_float3( src.scale.x, src.scale.y, src.scale.z );
        // only one skybox texture supported
        if( !src.environmentMapName.empty() && first )
        {
            dest.skyboxTextureId = m_demandTextureCache->createSkyboxTextureFromFile( src.environmentMapName );
            first = false;
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

OptixModule PbrtScene::createModule( const char* optixir, size_t optixirSize )
{
    const OptixModuleCompileOptions    compileOptions{ getCompileOptions() };
    const OptixPipelineCompileOptions& pipelineCompileOptions{ m_renderer->getPipelineCompileOptions() };

    OptixModule        module;
    OptixDeviceContext context = m_renderer->getDeviceContext();
    OTK_ERROR_CHECK_LOG( optixModuleCreate( context, &compileOptions, &pipelineCompileOptions, optixir, optixirSize,
                                            LOG, &LOG_SIZE, &module ) );
    return module;
}

void PbrtScene::createModules()
{
    const OptixModuleCompileOptions    compileOptions{ getCompileOptions() };
    const OptixPipelineCompileOptions& pipelineCompileOptions{ m_renderer->getPipelineCompileOptions() };

    OptixDeviceContext context          = m_renderer->getDeviceContext();
    auto               getBuiltinModule = [&]( OptixPrimitiveType type ) {
        OptixModule           module;
        OptixBuiltinISOptions builtinOptions{};
        builtinOptions.builtinISModuleType = type;
        builtinOptions.buildFlags          = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        OTK_ERROR_CHECK_LOG( optixBuiltinISModuleGet( context, &compileOptions, &pipelineCompileOptions, &builtinOptions, &module ) );
        return module;
    };
    m_sceneModule    = createModule( DemandPbrtSceneCudaText(), DemandPbrtSceneCudaSize );
    m_triangleModule = getBuiltinModule( OPTIX_PRIMITIVE_TYPE_TRIANGLE );
    m_sphereModule   = getBuiltinModule( OPTIX_PRIMITIVE_TYPE_SPHERE );
}

void PbrtScene::createProgramGroups()
{
    OptixProgramGroupOptions options{};
    m_programGroups.resize( +ProgramGroupIndex::NUM_STATIC_PROGRAM_GROUPS );
    OptixProgramGroupDesc descs[+ProgramGroupIndex::NUM_STATIC_PROGRAM_GROUPS]{};
    const char* const     proxyMaterialCHFunctionName = m_materialLoader->getCHFunctionName();
    otk::ProgramGroupDescBuilder( descs, m_sceneModule )
        .raygen( "__raygen__perspectiveCamera" )
        .miss( "__miss__backgroundColor" )
        .hitGroupISCH( m_sceneModule, m_geometryLoader->getISFunctionName(), m_sceneModule, m_geometryLoader->getCHFunctionName() )
        .hitGroupISCH( m_triangleModule, nullptr, m_sceneModule, proxyMaterialCHFunctionName )
        .hitGroupISAHCH( m_triangleModule, nullptr, m_sceneModule, "__anyhit__alphaCutOutPartialMesh", m_sceneModule, proxyMaterialCHFunctionName )
        .hitGroupISCH( m_sphereModule, nullptr, m_sceneModule, proxyMaterialCHFunctionName )
        .hitGroupISAHCH( m_sphereModule, nullptr, m_sceneModule, "__anyhit__sphere", m_sceneModule, proxyMaterialCHFunctionName );
    OptixDeviceContext context = m_renderer->getDeviceContext();
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, descs, m_programGroups.size(), &options, LOG, &LOG_SIZE,
                                                  m_programGroups.data() ) );
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
    m_topLevelInstances.push_back( instance );
}

void PbrtScene::createTopLevelTraversable( CUstream stream )
{
    m_topLevelInstances.copyToDeviceAsync( stream );
    if( m_options.sync )
    {
        OTK_CUDA_SYNC_CHECK();
    }

    const uint_t    NUM_BUILD_INPUTS = 1;
    OptixBuildInput inputs[NUM_BUILD_INPUTS]{};
    otk::BuildInputBuilder( inputs ).instanceArray( m_topLevelInstances, m_topLevelInstances.size() );

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

void PbrtScene::initialize( CUstream stream )
{
    std::cout << "Reading " << m_options.sceneFile << "...\n";
    m_scene = m_sceneLoader->parseFile( m_options.sceneFile );
    if( m_scene->errors > 0 )
        throw std::runtime_error( "Couldn't load scene " + m_options.sceneFile );

    std::cout << "\nScene " << m_options.sceneFile << " loaded: " << m_scene->freeShapes.size() << " free shapes, "
              << m_scene->objects.size() << " objects, " << m_scene->objectInstances.size() << " instances.\n";
    SceneProxyPtr proxy                = m_proxyFactory->scene( m_geometryLoader, m_scene );
    m_sceneProxies[proxy->getPageId()] = proxy;
    setCamera();

    createModules();
    createProgramGroups();
    m_renderer->setProgramGroups( m_programGroups );

    m_geometryLoader->setSbtIndex( +HitGroupIndex::PROXY_GEOMETRY );
    m_geometryLoader->copyToDeviceAsync( stream );
    OptixDeviceContext context = m_renderer->getDeviceContext();
    m_proxyInstanceTraversable = m_geometryLoader->createTraversable( context, stream );
    if( m_options.sync )
    {
        OTK_CUDA_SYNC_CHECK();
    }

    m_topLevelInstances.clear();
    pushInstance( m_proxyInstanceTraversable );
    createTopLevelTraversable( stream );
}

void PbrtScene::cleanup()
{
    for( OptixProgramGroup group : m_programGroups )
    {
        OTK_ERROR_CHECK( optixProgramGroupDestroy( group ) );
    }
    if( m_phongModule )
        OTK_ERROR_CHECK( optixModuleDestroy( m_phongModule ) );
    OTK_ERROR_CHECK( optixModuleDestroy( m_sceneModule ) );
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

    params.ambientColor        = make_float3( 0.4f, 0.4f, 0.4f );
    params.sceneEpsilon        = 0.005f;
    params.usePinholeCamera    = m_options.usePinholeCamera;
    params.traversable         = m_topLevelTraversable;
    params.demandGeomContext   = m_geometryLoader->getContext();
    const float3 yellow        = make_float3( 1.0f, 1.0f, 0.0 );
    params.demandMaterialColor = yellow;
    params.partialMaterials    = m_partialMaterials.typedDevicePtr();
    params.realizedMaterials   = m_realizedMaterials.typedDevicePtr();
    params.instanceNormals     = m_realizedNormals.typedDevicePtr();
    params.instanceUVs         = m_realizedUVs.typedDevicePtr();
    params.partialUVs          = m_partialUVs.typedDevicePtr();
    // Copy lights from the scene description; only need to do this once or if lights change interactively.
    if( params.numDirectionalLights == 0 && !m_scene->distantLights.empty() )
    {
        m_directionalLights.resize( m_scene->distantLights.size() );
        std::transform( m_scene->distantLights.begin(), m_scene->distantLights.end(), m_directionalLights.begin(),
                        []( const otk::pbrt::DistantLightDefinition& distant ) {
                            const pbrt::Vector3f direction( Normalize( distant.lightToWorld( distant.direction ) ) );
                            return DirectionalLight{ make_float3( direction.x, direction.y, direction.z ),
                                                     make_float3( distant.color.x * distant.scale.x,
                                                                  distant.color.y * distant.scale.y,
                                                                  distant.color.z * distant.scale.z ) };
                        } );
        m_directionalLights.copyToDevice();
        params.numDirectionalLights = static_cast<uint_t>( m_directionalLights.size() );
        params.directionalLights    = m_directionalLights.typedDevicePtr();
    }
    if( params.numInfiniteLights == 0 && !m_scene->infiniteLights.empty() )
    {
        realizeInfiniteLights();
        m_infiniteLights.copyToDevice();
        params.numInfiniteLights = static_cast<uint_t>( m_infiniteLights.size() );
        params.infiniteLights       = m_infiniteLights.typedDevicePtr();
    }

    m_demandLoader->launchPrepare( stream, params.demandContext );
}

void PbrtScene::resolveProxyGeometry( CUstream stream, uint_t proxyGeomId )
{
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
            std::cout << "Resolved proxy geometry " << proxyGeomId << " to " << IdRange{ subProxies } << '\n';
        }
    }
    else
    {
        // add instance to TLAS instances
        SceneGeometry geom;
        geom.instance                     = removedProxy->createGeometry( m_renderer->getDeviceContext(), stream );
        geom.materialId                   = m_materialLoader->add();
        geom.instance.instance.instanceId = geom.materialId;
        geom.instanceIndex                = m_topLevelInstances.size();
        m_topLevelInstances.push_back( geom.instance.instance );
        m_proxyMaterialGeometries[geom.materialId] = geom;
        if( m_options.verboseProxyGeometryResolution )
        {
            std::cout << "Resolved proxy geometry " << proxyGeomId << " to geometry instance " << geom.instanceIndex << '\n';
        }
        ++m_stats.numProxyMaterialsCreated;
        ++m_stats.numGeometriesRealized;
    }
    ++m_stats.numProxyGeometriesResolved;
}

inline float volume( const OptixAabb& bounds )
{
    return std::fabs( bounds.maxX - bounds.minX ) * std::fabs( bounds.maxY - bounds.minY )
           * std::fabs( bounds.maxZ - bounds.minZ );
}


std::vector<uint_t> PbrtScene::sortRequestedProxyGeometriesByCameraDistance()
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

template <typename Container>
void grow( Container& container, size_t size )
{
    if( container.size() < size )
    {
        container.resize( size );
    }
}

bool PbrtScene::resolveRequestedProxyGeometries( CUstream stream )
{
    if( m_options.oneShotGeometry && !m_resolveOneGeometry )
    {
        return false;
    }

    bool realized{};
    for( uint_t id : sortRequestedProxyGeometriesByCameraDistance() )
    {
        if( frameBudgetExceeded() )
        {
            break;
        }

        resolveProxyGeometry( stream, id );
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
        m_geometryLoader->copyToDeviceAsync( stream );
        OptixDeviceContext context               = m_renderer->getDeviceContext();
        m_proxyInstanceTraversable               = m_geometryLoader->createTraversable( context, stream );
        m_topLevelInstances[0].traversableHandle = m_proxyInstanceTraversable;
    }

    return realized;
}

uint_t PbrtScene::getTriangleRealizedMaterialSbtOffset( const GeometryInstance& instance )
{
    OptixDeviceContext       context = m_renderer->getDeviceContext();
    OptixProgramGroupOptions options{};
    OptixProgramGroup        group{};
    OptixProgramGroupDesc    groupDesc[1]{};

    // triangles with alpha map and diffuse map texture
    if( flagSet( instance.material.flags, MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP ) )
    {
        if( m_triangleAlphaDiffuseMapHitGroupIndex == 0 )
        {
            otk::ProgramGroupDescBuilder( groupDesc, m_sceneModule )             //
                .hitGroupISAHCH( m_triangleModule, nullptr,                      //
                                 m_sceneModule, "__anyhit__alphaCutOutMesh",     //
                                 m_sceneModule, "__closesthit__texturedMesh" );  //
            OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, groupDesc, 1, &options, LOG, &LOG_SIZE, &group ) );
            m_triangleAlphaDiffuseMapHitGroupIndex = m_programGroups.size() - +ProgramGroupIndex::HITGROUP_START;
            m_programGroups.push_back( group );
            m_renderer->setProgramGroups( m_programGroups );
        }
        return m_triangleAlphaDiffuseMapHitGroupIndex;
    }

    // triangles with alpha map texture
    if( flagSet( instance.material.flags, MaterialFlags::ALPHA_MAP ) )
    {
        if( m_triangleAlphaMapHitGroupIndex == 0 )
        {
            otk::ProgramGroupDescBuilder( groupDesc, m_sceneModule )          //
                .hitGroupISAHCH( m_triangleModule, nullptr,                   //
                                 m_sceneModule, "__anyhit__alphaCutOutMesh",  //
                                 m_phongModule, "__closesthit__mesh" );       //
            OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, groupDesc, 1, &options, LOG, &LOG_SIZE, &group ) );
            m_triangleAlphaMapHitGroupIndex = m_programGroups.size() - +ProgramGroupIndex::HITGROUP_START;
            m_programGroups.push_back( group );
            m_renderer->setProgramGroups( m_programGroups );
        }
        return m_triangleAlphaMapHitGroupIndex;
    }

    // triangles with diffuse map texture
    if( flagSet( instance.material.flags, MaterialFlags::DIFFUSE_MAP ) )
    {
        if( m_triangleDiffuseMapHitGroupIndex == 0 )
        {
            otk::ProgramGroupDescBuilder( groupDesc, m_sceneModule )           //
                .hitGroupISCH( m_triangleModule, nullptr,                      //
                               m_sceneModule, "__closesthit__texturedMesh" );  //
            OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, groupDesc, 1, &options, LOG, &LOG_SIZE, &group ) );
            m_triangleDiffuseMapHitGroupIndex = m_programGroups.size() - +ProgramGroupIndex::HITGROUP_START;
            m_programGroups.push_back( group );
            m_renderer->setProgramGroups( m_programGroups );
        }
        return m_triangleDiffuseMapHitGroupIndex;
    }

    // untextured triangles
    if( m_triangleHitGroupIndex == 0 )
    {
        otk::ProgramGroupDescBuilder( groupDesc, m_sceneModule )   //
            .hitGroupISCH( m_triangleModule, nullptr,              //
                           m_phongModule, "__closesthit__mesh" );  //
        OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, groupDesc, 1, &options, LOG, &LOG_SIZE, &group ) );
        m_triangleHitGroupIndex = m_programGroups.size() - +ProgramGroupIndex::HITGROUP_START;
        m_programGroups.push_back( group );
        m_renderer->setProgramGroups( m_programGroups );
    }
    return m_triangleHitGroupIndex;
}

uint_t PbrtScene::getSphereRealizedMaterialSbtOffset()
{
    // untextured sphere
    if( m_sphereHitGroupIndex == 0 )
    {
        const OptixDeviceContext context = m_renderer->getDeviceContext();
        OptixProgramGroupOptions options{};
        OptixProgramGroup        group{};
        OptixProgramGroupDesc    groupDesc[1]{};

        otk::ProgramGroupDescBuilder( groupDesc, m_sceneModule )
            .hitGroupISCH( m_sphereModule, nullptr, m_phongModule, "__closesthit__sphere" );
        OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, groupDesc, 1, &options, LOG, &LOG_SIZE, &group ) );
        m_sphereHitGroupIndex = m_programGroups.size() - +ProgramGroupIndex::HITGROUP_START;
        m_programGroups.push_back( group );
        m_renderer->setProgramGroups( m_programGroups );
    }
    return m_sphereHitGroupIndex;
}

uint_t PbrtScene::getRealizedMaterialSbtOffset( const GeometryInstance& instance )
{
    if( instance.primitive == GeometryPrimitive::TRIANGLE )
    {
        return getTriangleRealizedMaterialSbtOffset( instance );
    }
    if( instance.primitive == GeometryPrimitive::SPHERE )
    {
        return getSphereRealizedMaterialSbtOffset();
    }
    throw std::runtime_error( "Unimplemented primitive type " + std::to_string( +instance.primitive ) );
}

static demandLoading::TextureDescriptor getTextureDescriptor()
{
    demandLoading::TextureDescriptor desc{};
    desc.addressMode[0]   = CU_TR_ADDRESS_MODE_WRAP;
    desc.addressMode[1]   = CU_TR_ADDRESS_MODE_WRAP;
    desc.filterMode       = CU_TR_FILTER_MODE_POINT;
    desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    desc.maxAnisotropy    = 1;
    return desc;
}

MaterialResolution PbrtScene::resolveMaterial( uint_t proxyMaterialId )
{
    auto it = m_proxyMaterialGeometries.find( proxyMaterialId );
    if( it == m_proxyMaterialGeometries.end() )
    {
        throw std::runtime_error( "Unknown material id " + std::to_string( proxyMaterialId ) );
    }

    SceneGeometry& geom = it->second;
    if( geom.materialId != proxyMaterialId )
    {
        throw std::runtime_error( "Mismatched material id; expected " + std::to_string( geom.materialId ) + ", got "
                                  + std::to_string( proxyMaterialId ) );
    }

    // Only triangle meshes support alpha maps currently.
    // TODO: support alpha maps on spheres
    if( geom.instance.primitive == GeometryPrimitive::TRIANGLE )
    {
        // phase 1 alpha map resolution
        if( flagSet( geom.instance.material.flags, MaterialFlags::ALPHA_MAP )
            && !flagSet( geom.instance.material.flags, MaterialFlags::ALPHA_MAP_ALLOCATED ) )
        {
            geom.instance.material.alphaTextureId =
                m_demandTextureCache->createAlphaTextureFromFile( geom.instance.alphaMapFileName );
            geom.instance.material.flags |= MaterialFlags::ALPHA_MAP_ALLOCATED;
            const size_t numProxyMaterials = proxyMaterialId + 1;  // ids are zero based
            grow( m_partialMaterials, numProxyMaterials );
            grow( m_partialUVs, numProxyMaterials );
            m_partialMaterials[proxyMaterialId].alphaTextureId = geom.instance.material.alphaTextureId;
            m_partialUVs[proxyMaterialId]                      = geom.instance.uvs;
            m_topLevelInstances[geom.instanceIndex].sbtOffset  = +HitGroupIndex::PROXY_MATERIAL_TRIANGLE_ALPHA;
            if( m_options.verboseProxyMaterialResolution )
            {
                std::cout << "Resolved proxy material " << proxyMaterialId << " to partial alpha texture "
                          << geom.instance.material.alphaTextureId << '\n';
            }
            return MaterialResolution::PARTIAL;
        }

        // phase 2 alpha map resolution
        if( flagSet( geom.instance.material.flags, MaterialFlags::ALPHA_MAP_ALLOCATED ) )
        {
            // not strictly necessary, but indicates this partial material has been resolved completely
            m_partialMaterials[proxyMaterialId].alphaTextureId = 0;
            m_partialUVs[proxyMaterialId]                      = nullptr;
        }

        // diffuse map resolution
        if( flagSet( geom.instance.material.flags, MaterialFlags::DIFFUSE_MAP )
            && !flagSet( geom.instance.material.flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) )
        {
            geom.instance.material.diffuseTextureId =
                m_demandTextureCache->createDiffuseTextureFromFile( geom.instance.diffuseMapFileName );
            geom.instance.material.flags |= MaterialFlags::DIFFUSE_MAP_ALLOCATED;
        }
    }

    if( m_phongModule == nullptr )
    {
        m_phongModule = createModule( PhongMaterialCudaText(), PhongMaterialCudaSize );
    }

    geom.instance.instance.sbtOffset  = getRealizedMaterialSbtOffset( geom.instance );
    geom.instance.instance.instanceId = m_realizedMaterials.size();
    m_realizedMaterials.push_back( geom.instance.material );
    m_realizedNormals.push_back( geom.instance.normals );
    m_realizedUVs.push_back( geom.instance.uvs );
    m_topLevelInstances[geom.instanceIndex]  = geom.instance.instance;
    m_realizedGeometries[geom.instanceIndex] = geom;
    if( m_options.verboseProxyMaterialResolution )
    {
        std::cout
            << "Resolved proxy material " << proxyMaterialId << " for instance " << geom.instance.instance.instanceId
            << ( flagSet( geom.instance.material.flags, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) ? " with diffuse map" : "" )
            << '\n';
    }
    m_materialLoader->remove( proxyMaterialId );
    m_proxyMaterialGeometries.erase( proxyMaterialId );
    return MaterialResolution::FULL;
}

MaterialResolution PbrtScene::resolveRequestedProxyMaterials( CUstream stream )
{
    if( m_options.oneShotMaterial && !m_resolveOneMaterial )
    {
        return MaterialResolution::NONE;
    }

    MaterialResolution realized{ MaterialResolution::NONE };
    for( uint_t id : m_materialLoader->requestedMaterialIds() )
    {
        if( frameBudgetExceeded() )
        {
            break;
        }

        realized = std::max( realized, resolveMaterial( id ) );

        if( m_resolveOneMaterial )
        {
            m_resolveOneMaterial = false;
            break;
        }
    }

    switch( realized )
    {
        case MaterialResolution::NONE:
            break;
        case MaterialResolution::PARTIAL:
            m_partialMaterials.copyToDeviceAsync( stream );
            m_partialUVs.copyToDeviceAsync( stream );
            ++m_stats.numPartialMaterialsRealized;
            break;
        case MaterialResolution::FULL:
            m_partialMaterials.copyToDeviceAsync( stream );
            m_partialUVs.copyToDeviceAsync( stream );
            m_realizedNormals.copyToDeviceAsync( stream );
            m_realizedUVs.copyToDeviceAsync( stream );
            m_realizedMaterials.copyToDeviceAsync( stream );
            m_materialLoader->clearRequestedMaterialIds();
            ++m_stats.numMaterialsRealized;
            break;
    }
    return realized;
}

bool PbrtScene::frameBudgetExceeded() const
{
    // infinite frame budget when rendering to a file
    if( !m_interactive )
    {
        return false;
    }

    const Clock::duration duration = Clock::now() - m_frameStart;
    return duration > m_frameTime;
}

bool PbrtScene::beforeLaunch( CUstream stream, Params& params )
{
    m_ticket.wait();
    m_frameStart = Clock::now();

    const bool               realizedGeometry = resolveRequestedProxyGeometries( stream );
    const MaterialResolution realizedMaterial = resolveRequestedProxyMaterials( stream );

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

ScenePtr createPbrtScene( const Options&        options,
                          PbrtSceneLoaderPtr    pbrt,
                          DemandTextureCachePtr demandTextureCache,
                          ProxyFactoryPtr       proxyFactory,
                          DemandLoaderPtr       demandLoader,
                          GeometryLoaderPtr     geometryLoader,
                          MaterialLoaderPtr     materialLoader,
                          RendererPtr           renderer )
{
    return std::make_shared<PbrtScene>( options, std::move( pbrt ), std::move( demandTextureCache ),
                                        std::move( proxyFactory ), std::move( demandLoader ),
                                        std::move( geometryLoader ), std::move( materialLoader ), std::move( renderer ) );
}

}  // namespace demandPbrtScene
