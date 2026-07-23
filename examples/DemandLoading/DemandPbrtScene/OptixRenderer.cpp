// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/OptixRenderer.h"

#include "DemandPbrtScene/Conversions.h"
#include "DemandPbrtScene/Options.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/Scene.h"

#include <OptiXToolkit/Error/cuErrorCheck.h>
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
#include <stdexcept>
#include <utility>
#include <vector>

namespace demandPbrtScene {

OptixRenderer::PipelineState::PipelineState( OptixPipeline                         pipeline_,
                                             std::vector<OptixProgramGroup>        programGroups_,
                                             std::vector<OptixProgramGroup>        callableProgramGroups_ )
    : pipeline( pipeline_ )
    , programGroups( std::move( programGroups_ ) )
    , callableProgramGroups( std::move( callableProgramGroups_ ) )
    , rayGenRecord( 1 )
    , missRecord( 1 )
    , hitGroupRecords( +ProgramGroupIndex::NUM_STATIC_PROGRAM_GROUPS )
{
}

OptixRenderer::PipelineState::~PipelineState()
{
    if( launchCompleteEvent )
    {
        OTK_ERROR_CHECK_NOTHROW( cuEventDestroy( launchCompleteEvent ) );
    }
    if( pipeline )
    {
        OTK_ERROR_CHECK_NOTHROW( optixPipelineDestroy( pipeline ) );
    }
}

void OptixRenderer::PipelineState::recordLaunchCompleteEvent( CUstream stream )
{
    if( !launchCompleteEvent )
    {
        OTK_ERROR_CHECK( cuEventCreate( &launchCompleteEvent, CU_EVENT_DISABLE_TIMING ) );
    }
    OTK_ERROR_CHECK( cuEventRecord( launchCompleteEvent, stream ) );
}

bool OptixRenderer::PipelineState::launchComplete() const
{
    if( !launchCompleteEvent )
    {
        return true;
    }

    const CUresult result = cuEventQuery( launchCompleteEvent );
    if( result == CUDA_SUCCESS )
    {
        return true;
    }
    if( result == CUDA_ERROR_NOT_READY )
    {
        return false;
    }
    OTK_ERROR_CHECK( result );
    return false;
}

void OptixRenderer::PipelineState::waitForLaunchComplete() const
{
    if( launchCompleteEvent )
    {
        OTK_ERROR_CHECK( cuEventSynchronize( launchCompleteEvent ) );
    }
}

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
    m_pendingState.reset();
    m_programGroups   = value;
    m_pipelineChanged = true;
    m_sbtChanged      = true;
}

void OptixRenderer::setCallableProgramGroups( const std::vector<OptixProgramGroup>& value )
{
    m_pendingState.reset();
    m_callableProgramGroups = value;
    m_pipelineChanged       = true;
    m_sbtChanged            = true;
}

#ifdef OTK_USE_MDL
void OptixRenderer::setPipelineState( OptixPipeline                         pipeline,
                                      const std::vector<OptixProgramGroup>& programGroups,
                                      const std::vector<OptixProgramGroup>& callableProgramGroups )
{
    m_pendingState          = std::make_shared<PipelineState>( pipeline, programGroups, callableProgramGroups );
    m_programGroups         = programGroups;
    m_callableProgramGroups = callableProgramGroups;
    m_pipelineChanged       = false;
    m_sbtChanged            = true;
    setClearAccumulator();
}
#endif

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

const int NUM_PAYLOAD_VALUES   = 3;
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

std::shared_ptr<OptixRenderer::PipelineState> OptixRenderer::createPipelineState()
{
    const uint_t             maxTraceDepth{ 1 };
    OptixPipeline            pipeline{};
    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = maxTraceDepth;
    std::vector<OptixProgramGroup> pipelineProgramGroups{ m_programGroups };
    std::copy( m_callableProgramGroups.cbegin(), m_callableProgramGroups.cend(), std::back_inserter( pipelineProgramGroups ) );
    OTK_ERROR_CHECK_LOG( optixPipelineCreate( m_context, &m_pipelineCompileOptions, &linkOptions, pipelineProgramGroups.data(),
                                              pipelineProgramGroups.size(), LOG, &LOG_SIZE, &pipeline ) );

    OptixStackSizes stackSizes{};
    for( OptixProgramGroup group : pipelineProgramGroups )
    {
#if OPTIX_VERSION < 70700
        OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( group, &stackSizes ) );
#else
        OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( group, &stackSizes, pipeline ) );
#endif
    }
    uint_t       directCallableTraversalStackSize{};
    uint_t       directCallableStateStackSize{};
    uint_t       continuationStackSize{};
    const uint_t maxDirectCallableDepth{ m_callableProgramGroups.empty() ? 0U : 1U };
    OTK_ERROR_CHECK( optixUtilComputeStackSizes( &stackSizes, maxTraceDepth, 0, maxDirectCallableDepth, &directCallableTraversalStackSize,
                                                 &directCallableStateStackSize, &continuationStackSize ) );
    const uint_t maxTraversableDepth = 3;
    OTK_ERROR_CHECK( optixPipelineSetStackSize( pipeline, directCallableTraversalStackSize, directCallableStateStackSize,
                                                continuationStackSize, maxTraversableDepth ) );
    return std::make_shared<PipelineState>( pipeline, m_programGroups, m_callableProgramGroups );
}

void OptixRenderer::writeRayGenRecords( CUstream stream, PipelineState& state )
{
    // A single raygen record.
    state.rayGenRecord.packHeader( 0, state.programGroups[+ProgramGroupIndex::RAYGEN] );
    state.rayGenRecord.copyToDeviceAsync( stream );
}

void OptixRenderer::writeMissRecords( CUstream stream, PipelineState& state )
{
    // A single miss record.
    state.missRecord.packHeader( 0, state.programGroups[+ProgramGroupIndex::MISS] );
    state.missRecord.copyToDeviceAsync( stream );
}

void OptixRenderer::writeHitGroupRecords( CUstream stream, PipelineState& state )
{
    auto packHeader = [&]( HitGroupIndex hitGroup, ProgramGroupIndex programGroup ) {
        state.hitGroupRecords.packHeader( +hitGroup, state.programGroups[+programGroup] );
    };
    packHeader( HitGroupIndex::PROXY_GEOMETRY, ProgramGroupIndex::HITGROUP_PROXY_GEOMETRY );
    packHeader( HitGroupIndex::PROXY_MATERIAL_TRIANGLE, ProgramGroupIndex::HITGROUP_PROXY_MATERIAL_TRIANGLE );
    packHeader( HitGroupIndex::PROXY_MATERIAL_TRIANGLE_ALPHA, ProgramGroupIndex::HITGROUP_PROXY_MATERIAL_TRIANGLE_ALPHA );
    packHeader( HitGroupIndex::PROXY_MATERIAL_SPHERE, ProgramGroupIndex::HITGROUP_PROXY_MATERIAL_SPHERE );
    packHeader( HitGroupIndex::PROXY_MATERIAL_SPHERE_ALPHA, ProgramGroupIndex::HITGROUP_PROXY_MATERIAL_SPHERE_ALPHA );

    // Initially no hitgroup record(s) for realized materials.
    const size_t count = state.programGroups.size() - +ProgramGroupIndex::NUM_STATIC_PROGRAM_GROUPS;
    state.hitGroupRecords.resize( +ProgramGroupIndex::NUM_STATIC_PROGRAM_GROUPS + count );
    for( size_t i = 0; i < count; ++i )
    {
        state.hitGroupRecords.packHeader( +HitGroupIndex::REALIZED_MATERIAL_START + i,
                                          state.programGroups[+ProgramGroupIndex::HITGROUP_REALIZED_MATERIAL_START + i] );
    }

    state.hitGroupRecords.copyToDeviceAsync( stream );
}

void OptixRenderer::writeCallableRecords( CUstream stream, PipelineState& state )
{
    state.callableRecords.resize( state.callableProgramGroups.size() );
    for( size_t i = 0; i < state.callableProgramGroups.size(); ++i )
    {
        state.callableRecords.packHeader( i, state.callableProgramGroups[i] );
    }
    if( !state.callableProgramGroups.empty() )
    {
        state.callableRecords.copyToDeviceAsync( stream );
    }
}

void OptixRenderer::writeSbt( PipelineState& state )
{
    state.sbt.raygenRecord                 = state.rayGenRecord;
    state.sbt.missRecordBase               = state.missRecord;
    state.sbt.missRecordStrideInBytes      = toUInt( sizeof( otk::Record<otk::EmptyData> ) );
    state.sbt.missRecordCount              = containerSize( state.missRecord );
    state.sbt.hitgroupRecordBase           = state.hitGroupRecords;
    state.sbt.hitgroupRecordCount          = containerSize( state.hitGroupRecords );
    state.sbt.hitgroupRecordStrideInBytes  = toUInt( sizeof( otk::Record<otk::EmptyData> ) );
    state.sbt.callablesRecordBase          = state.callableRecords;
    state.sbt.callablesRecordCount         = containerSize( state.callableRecords );
    state.sbt.callablesRecordStrideInBytes = toUInt( sizeof( otk::Record<otk::EmptyData> ) );
}

void OptixRenderer::buildShaderBindingTable( CUstream stream, PipelineState& state )
{
    writeRayGenRecords( stream, state );
    writeMissRecords( stream, state );
    writeHitGroupRecords( stream, state );
    writeCallableRecords( stream, state );
    writeSbt( state );
}

void OptixRenderer::activatePendingPipelineState()
{
    std::shared_ptr<PipelineState> retiredState{ std::move( m_activeState ) };
    m_activeState = m_pendingState;
    m_pendingState.reset();
    retirePipelineState( std::move( retiredState ) );
}

void OptixRenderer::retirePipelineState( std::shared_ptr<PipelineState> state )
{
    if( !state || state->launchComplete() )
    {
        return;
    }
    m_retiredStates.push_back( std::move( state ) );
}

void OptixRenderer::collectRetiredPipelineStates()
{
    m_retiredStates.erase( std::remove_if( m_retiredStates.begin(), m_retiredStates.end(),
                                           []( const std::shared_ptr<PipelineState>& state ) {
                                               return state->launchComplete();
                                           } ),
                           m_retiredStates.end() );
}

void OptixRenderer::waitForPipelineStates()
{
    for( const std::shared_ptr<PipelineState>& state : m_retiredStates )
    {
        state->waitForLaunchComplete();
    }
    if( m_activeState )
    {
        m_activeState->waitForLaunchComplete();
    }
}

void OptixRenderer::cleanup()
{
    waitForPipelineStates();
    m_retiredStates.clear();
    m_pendingState.reset();
    m_activeState.reset();
    OTK_ERROR_CHECK( optixDeviceContextDestroy( m_context ) );
}

void OptixRenderer::beforeLaunch( CUstream stream )
{
    collectRetiredPipelineStates();

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
        m_pendingState    = createPipelineState();
        m_pipelineChanged = false;
        m_sbtChanged      = true;
    }
    if( m_pendingState )
    {
        buildShaderBindingTable( stream, *m_pendingState );
        activatePendingPipelineState();
        m_sbtChanged = false;
    }
    else if( m_sbtChanged && m_activeState )
    {
        buildShaderBindingTable( stream, *m_activeState );
        m_sbtChanged = false;
    }
}

void OptixRenderer::launch( CUstream stream, uchar4* image )
{
    const std::shared_ptr<PipelineState> state{ m_activeState };
    if( !state )
    {
        throw std::runtime_error( "Cannot launch before an OptiX pipeline state is active" );
    }
    m_accumulator.resize( m_options.width, m_options.height );
    if( m_clearAccumulator )
    {
        m_accumulator.clear();
        m_clearAccumulator = false;
    }
    m_params[0].image       = image;
    m_params[0].accumulator = m_accumulator.getBuffer();
    m_params[0].renderMode  = m_options.renderMode;
    m_params.copyToDevice();
    OTK_ERROR_CHECK( optixLaunch( state->pipeline, stream, m_params, sizeof( Params ), &state->sbt, m_options.width,
                                  m_options.height, /*depth=*/1 ) );
    state->recordLaunchCompleteEvent( stream );
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
