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

#include "OptixRenderer.h"

#include "Options.h"
#include "Params.h"
#include "Scene.h"

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/Util/Logger.h>

#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

namespace demandPbrtScene {

OptixRenderer::OptixRenderer( const Options& options, int numAttributes )
    : m_options( options )
    , m_numAttributes( numAttributes )
{
}

void OptixRenderer::initialize( CUstream stream )
{
    createOptixContext();
    initPipelineOpts();
    initializeParamsFromOptions();
}

void OptixRenderer::setProgramGroups( const std::vector<OptixProgramGroup>& value )
{
    m_programGroups   = value;
    m_pipelineChanged = true;
    m_sbtChanged      = true;
}

void OptixRenderer::createOptixContext()
{
    CUcontext                 cuCtx{};  // zero means take the current context
    OptixDeviceContextOptions options{};
    otk::util::setLogger( options );
#ifndef NDEBUG
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;    
#endif
    OTK_ERROR_CHECK( optixDeviceContextCreate( cuCtx, &options, &m_context ) );
}

const int NUM_PAYLOAD_VALUES   = 7;
const int NUM_ATTRIBUTE_VALUES = 3;

void OptixRenderer::initPipelineOpts()
{
    m_pipelineCompileOptions.usesMotionBlur        = 0;
    m_pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    m_pipelineCompileOptions.numPayloadValues      = NUM_PAYLOAD_VALUES;
    m_pipelineCompileOptions.numAttributeValues    = std::max( NUM_ATTRIBUTE_VALUES, m_numAttributes );
    m_pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    m_pipelineCompileOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    m_pipelineCompileOptions.pipelineLaunchParamsVariableName = PARAMS_STRING_NAME;
}

void OptixRenderer::initializeParamsFromOptions()
{
    m_params[0].debug.enabled = m_options.debug;
    if( m_options.debug )
    {
        m_params[0].debug.debugIndexSet = true;
        m_params[0].debug.debugIndex    = make_uint3( m_options.debugPixel.x, m_options.debugPixel.y, 0 );
    }
    m_params[0].useFaceForward = m_options.faceForward;
}

void OptixRenderer::createPipeline()
{
    const uint_t             maxTraceDepth = 1;
    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = maxTraceDepth;
    OTK_ERROR_CHECK_LOG( optixPipelineCreate( m_context, &m_pipelineCompileOptions, &linkOptions, m_programGroups.data(),
                                              m_programGroups.size(), LOG, &LOG_SIZE, &m_pipeline ) );

    OptixStackSizes stackSizes{};
    for( OptixProgramGroup group : m_programGroups )
    {
#if OPTIX_VERSION < 70700
        OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( group, &stackSizes ) );
#else
        OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( group, &stackSizes, m_pipeline ) );
#endif
    }
    uint_t directCallableTraversalStackSize{};
    uint_t directCallableStateStackSize{};
    uint_t continuationStackSize{};
    OTK_ERROR_CHECK( optixUtilComputeStackSizes( &stackSizes, maxTraceDepth, 0, 0, &directCallableTraversalStackSize,
                                                 &directCallableStateStackSize, &continuationStackSize ) );
    const uint_t maxTraversableDepth = 3;
    OTK_ERROR_CHECK( optixPipelineSetStackSize( m_pipeline, directCallableTraversalStackSize, directCallableStateStackSize,
                                                continuationStackSize, maxTraversableDepth ) );
}

void OptixRenderer::writeRayGenRecords( CUstream stream )
{
    // A single raygen record.
    m_rayGenRecord.packHeader( 0, m_programGroups[+ProgramGroupIndex::RAYGEN] );
    m_rayGenRecord.copyToDeviceAsync( stream );
}

void OptixRenderer::writeMissRecords( CUstream stream )
{
    // A single miss record.
    m_missRecord.packHeader( 0, m_programGroups[+ProgramGroupIndex::MISS] );
    m_missRecord.copyToDeviceAsync( stream );
}

void OptixRenderer::writeHitGroupRecords( CUstream stream )
{
    auto packHeader = [&]( HitGroupIndex hitGroup, ProgramGroupIndex programGroup ) {
        m_hitGroupRecords.packHeader( +hitGroup, m_programGroups[+programGroup] );
    };
    packHeader( HitGroupIndex::PROXY_GEOMETRY, ProgramGroupIndex::HITGROUP_PROXY_GEOMETRY );
    packHeader( HitGroupIndex::PROXY_MATERIAL_TRIANGLE, ProgramGroupIndex::HITGROUP_PROXY_MATERIAL_TRIANGLE );
    packHeader( HitGroupIndex::PROXY_MATERIAL_TRIANGLE_ALPHA, ProgramGroupIndex::HITGROUP_PROXY_MATERIAL_TRIANGLE_ALPHA );
    packHeader( HitGroupIndex::PROXY_MATERIAL_SPHERE, ProgramGroupIndex::HITGROUP_PROXY_MATERIAL_SPHERE );
    packHeader( HitGroupIndex::PROXY_MATERIAL_SPHERE_ALPHA, ProgramGroupIndex::HITGROUP_PROXY_MATERIAL_SPHERE_ALPHA );

    // Initially no hitgroup record(s) for realized materials.
    const size_t count = m_programGroups.size() - +ProgramGroupIndex::NUM_STATIC_PROGRAM_GROUPS;
    m_hitGroupRecords.resize( +ProgramGroupIndex::NUM_STATIC_PROGRAM_GROUPS + count );
    for( size_t i = 0; i < count; ++i )
    {
        m_hitGroupRecords.packHeader( +HitGroupIndex::REALIZED_MATERIAL_START + i,
                                      m_programGroups[+ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + i] );
    }

    m_hitGroupRecords.copyToDeviceAsync( stream );
}

void OptixRenderer::writeSbt()
{
    m_sbt.raygenRecord                = m_rayGenRecord;
    m_sbt.missRecordBase              = m_missRecord;
    m_sbt.missRecordStrideInBytes     = static_cast<uint_t>( sizeof( otk::Record<otk::EmptyData> ) );
    m_sbt.missRecordCount             = static_cast<uint_t>( m_missRecord.size() );
    m_sbt.hitgroupRecordBase          = m_hitGroupRecords;
    m_sbt.hitgroupRecordCount         = static_cast<uint_t>( m_hitGroupRecords.size() );
    m_sbt.hitgroupRecordStrideInBytes = static_cast<uint_t>( sizeof( otk::Record<otk::EmptyData> ) );
}

void OptixRenderer::buildShaderBindingTable( CUstream stream )
{
    writeRayGenRecords( stream );
    writeMissRecords( stream );
    writeHitGroupRecords( stream );
    writeSbt();
}

void OptixRenderer::cleanup()
{
    if( m_pipeline )
    {
        OTK_ERROR_CHECK( optixPipelineDestroy( m_pipeline ) );
    }
    OTK_ERROR_CHECK( optixDeviceContextDestroy( m_context ) );
}

void OptixRenderer::beforeLaunch( CUstream stream )
{
    Params& params    = m_params[0];
    params.width      = m_options.width;
    params.height     = m_options.height;
    params.background = m_options.background;
    if( m_params[0].debug.enabled )
    {
        if( m_options.oneShotDebug )
        {
            if( m_fireOneDebugDump )
            {
                params.debug.dumpSuppressed = false;
                m_fireOneDebugDump          = false;
            }
        }
        else
        {
            params.debug.dumpSuppressed = false;
        }
    }
    params.useFaceForward = m_options.faceForward;

    if( m_pipelineChanged )
    {
        createPipeline();
        m_pipelineChanged = false;
    }
    if( m_sbtChanged )
    {
        buildShaderBindingTable( stream );
        m_sbtChanged = false;
    }
}

void OptixRenderer::launch( CUstream stream, uchar4* image )
{
    m_accumulator.resize( m_options.width, m_options.height );
    if( m_clearAccumulator )
    {
        m_accumulator.clear();
        m_clearAccumulator = false;
    }
    m_params[0].image = image;
    m_params[0].accumulator = m_accumulator.getBuffer();
    m_params[0].renderMode = m_options.renderMode;
    m_params.copyToDevice();
    OTK_ERROR_CHECK( optixLaunch( m_pipeline, stream, m_params, sizeof( Params ), &m_sbt, m_options.width,
                                  m_options.height, /*depth=*/1 ) );
    if( m_options.sync )
    {
        OTK_CUDA_SYNC_CHECK();
    }
}

void OptixRenderer::afterLaunch()
{
    if( m_params[0].debug.enabled && m_params[0].debug.debugIndexSet && m_options.oneShotDebug )
    {
        m_params[0].debug.dumpSuppressed = true;
    }
}

void OptixRenderer::fireOneDebugDump()
{
    m_fireOneDebugDump = true;
}

RendererPtr createRenderer( const Options& options, int numAttributes )
{
    return std::make_shared<OptixRenderer>( options, numAttributes );
}

}  // namespace demandPbrtScene
