// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/ProgramGroups.h"

#include "DemandPbrtScene/DemandTextureCache.h"
#include "DemandPbrtScene/IdRangePrinter.h"
#include "DemandPbrtScene/ImageSourceFactory.h"
#include "DemandPbrtScene/Options.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/Renderer.h"
#include "DemandPbrtScene/SceneAdapters.h"
#include "DemandPbrtScene/SceneProxy.h"
#include "DemandPbrtScene/Stopwatch.h"

#include <DemandPbrtSceneKernelCuda.h>

#include <OptiXToolkit/DemandGeometry/GeometryLoader.h>
#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <OptiXToolkit/OptiXMemory/CompileOptions.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneLoader.h>

#include <optix_stubs.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif

namespace demandPbrtScene {

static OptixModuleCompileOptions getCompileOptions()
{
    OptixModuleCompileOptions compileOptions{};
    compileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    otk::configModuleCompileOptions( compileOptions );

    return compileOptions;
}

namespace {

class PbrtProgramGroups : public ProgramGroups
{
  public:
    PbrtProgramGroups( GeometryLoaderPtr geometryLoader, MaterialLoaderPtr materialLoader, RendererPtr renderer )
        : m_geometryLoader( std::move( geometryLoader ) )
        , m_materialLoader( std::move( materialLoader ) )
        , m_renderer( std::move( renderer ) )
    {
    }

    void initialize() override;
    void cleanup() override;

    uint_t getRealizedMaterialSbtOffset( const GeometryInstance& instance ) override;

  private:
    OptixModule createModule( const char* optixir, size_t optixirSize );
    void        createModules();
    void        createProgramGroups();
    uint_t      getTriangleRealizedMaterialSbtOffset( MaterialFlags flags );
    uint_t      getSphereRealizedMaterialSbtOffset();

    // Dependencies
    GeometryLoaderPtr m_geometryLoader;
    MaterialLoaderPtr m_materialLoader;
    RendererPtr       m_renderer;

    OptixModule                    m_sceneModule{};
    OptixModule                    m_phongModule{};
    OptixModule                    m_triangleModule{};
    OptixModule                    m_sphereModule{};
    std::vector<OptixProgramGroup> m_programGroups;
    size_t                         m_triangleHitGroupIndex{};
    size_t                         m_triangleAlphaMapHitGroupIndex{};
    size_t                         m_triangleDiffuseMapHitGroupIndex{};
    size_t                         m_triangleAlphaDiffuseMapHitGroupIndex{};
    size_t                         m_sphereHitGroupIndex{};
};

OptixModule PbrtProgramGroups::createModule( const char* optixir, size_t optixirSize )
{
    const OptixModuleCompileOptions    compileOptions{ getCompileOptions() };
    const OptixPipelineCompileOptions& pipelineCompileOptions{ m_renderer->getPipelineCompileOptions() };

    OptixModule        module;
    OptixDeviceContext context = m_renderer->getDeviceContext();
    OTK_ERROR_CHECK_LOG( optixModuleCreate( context, &compileOptions, &pipelineCompileOptions, optixir, optixirSize,
                                            LOG, &LOG_SIZE, &module ) );
    return module;
}

void PbrtProgramGroups::createModules()
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

void PbrtProgramGroups::createProgramGroups()
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

void PbrtProgramGroups::initialize()
{
    createModules();
    createProgramGroups();
    m_renderer->setProgramGroups( m_programGroups );
}

void PbrtProgramGroups::cleanup()
{
    for( OptixProgramGroup group : m_programGroups )
    {
        OTK_ERROR_CHECK( optixProgramGroupDestroy( group ) );
    }
    if( m_phongModule )
    {
        OTK_ERROR_CHECK( optixModuleDestroy( m_phongModule ) );
    }
    OTK_ERROR_CHECK( optixModuleDestroy( m_sceneModule ) );
}

uint_t PbrtProgramGroups::getTriangleRealizedMaterialSbtOffset( MaterialFlags flags )
{
    OptixDeviceContext       context = m_renderer->getDeviceContext();
    OptixProgramGroupOptions options{};
    OptixProgramGroup        group{};
    OptixProgramGroupDesc    groupDesc[1]{};

    // triangles with alpha map and diffuse map texture
    if( flagSet( flags, MaterialFlags::ALPHA_MAP | MaterialFlags::DIFFUSE_MAP ) )
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
    if( flagSet( flags, MaterialFlags::ALPHA_MAP ) )
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
    if( flagSet( flags, MaterialFlags::DIFFUSE_MAP ) )
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

uint_t PbrtProgramGroups::getSphereRealizedMaterialSbtOffset()
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

uint_t PbrtProgramGroups::getRealizedMaterialSbtOffset( const GeometryInstance& instance )
{
    if( m_phongModule == nullptr )
    {
        m_phongModule = createModule( PhongMaterialCudaText(), PhongMaterialCudaSize );
    }

    if( instance.primitive == GeometryPrimitive::TRIANGLE )
    {
        return getTriangleRealizedMaterialSbtOffset( instance.groups[0].material.flags );
    }
    if( instance.primitive == GeometryPrimitive::SPHERE )
    {
        return getSphereRealizedMaterialSbtOffset();
    }
    throw std::runtime_error( "Unimplemented primitive type " + std::to_string( +instance.primitive ) );
}

}  // namespace

ProgramGroupsPtr createProgramGroups( GeometryLoaderPtr geometryLoader, MaterialLoaderPtr materialLoader, RendererPtr renderer )
{
    return std::make_shared<PbrtProgramGroups>( std::move( geometryLoader ), std::move( materialLoader ), std::move( renderer ) );
}

}  // namespace demandPbrtScene
