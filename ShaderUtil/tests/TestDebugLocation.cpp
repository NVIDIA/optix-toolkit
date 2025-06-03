// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ShaderUtil/DebugLocation.h>

#include "TestDebugLocation.h"
#include "TestDebugLocationParams.h"
#include "TestShaderUtilsCuda.h"

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/Memory/DeviceBuffer.h>
#include <OptiXToolkit/Memory/SyncVector.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <OptiXToolkit/OptiXMemory/CompileOptions.h>
#include <OptiXToolkit/OptiXMemory/SyncRecord.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/vec_printers.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda.h>
#include <vector_types.h>

#include <gtest/gtest.h>

#include <functional>
#include <iomanip>
#include <sstream>
#include <vector>

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif

using namespace otk::shaderUtil::testing;

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif

namespace {

class Context
{
  public:
    Context() = default;
    Context( CUcontext fromContext, const OptixDeviceContextOptions* options )
    {
        OTK_ERROR_CHECK( optixDeviceContextCreate( fromContext, options, &m_context ) );
    }
    Context( const Context& rhs ) = delete;
    Context( Context&& rhs )      = delete;
    ~Context()
    {
        if( m_context )
            optixDeviceContextDestroy( m_context );
    }
    Context& operator=( const Context& rhs ) = delete;
    Context& operator=( Context&& rhs ) noexcept
    {
        m_context     = rhs.m_context;
        rhs.m_context = OptixDeviceContext{};
        return *this;
    }

    operator OptixDeviceContext() const { return m_context; }

  private:
    OptixDeviceContext m_context{};
};

class Module
{
  public:
    Module() = default;
    Module( OptixDeviceContext                 context,
            const OptixModuleCompileOptions*   moduleCompileOptions,
            const OptixPipelineCompileOptions* pipelineCompileOptions,
            const char*                        optixir,
            size_t                             optixirSize )
    {
        OTK_ERROR_CHECK_LOG( optixModuleCreate( context, moduleCompileOptions, pipelineCompileOptions, optixir,
                                                 optixirSize, LOG, &LOG_SIZE, &m_module ) );
    }
    Module( const Module& rhs ) = delete;
    Module( Module&& rhs )      = delete;
    ~Module()
    {
        if( m_module )
            optixModuleDestroy( m_module );
    }

    Module& operator=( const Module& rhs ) = delete;
    Module& operator=( Module&& rhs )
    {
        m_module     = rhs.m_module;
        rhs.m_module = OptixModule{};
        return *this;
    }

    operator OptixModule() { return m_module; }

  private:
    OptixModule m_module{};
};

class ProgramGroups
{
  public:
    ProgramGroups() = default;
    ProgramGroups( OptixDeviceContext              context,
                   const OptixProgramGroupDesc*    programDescriptions,
                   unsigned int                    numProgramGroups,
                   const OptixProgramGroupOptions* options )
    {
        m_programGroups.resize( numProgramGroups );
        OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, programDescriptions, numProgramGroups, options, LOG,
                                                       &LOG_SIZE, m_programGroups.data() ) );
    }
    ProgramGroups( const ProgramGroups& rhs ) = delete;
    ProgramGroups( ProgramGroups&& rhs )      = delete;
    ~ProgramGroups()
    {
        for( OptixProgramGroup group : m_programGroups )
            optixProgramGroupDestroy( group );
    }
    ProgramGroups& operator=( const ProgramGroups& rhs ) = delete;
    ProgramGroups& operator=( ProgramGroups&& rhs )
    {
        m_programGroups = std::move( rhs.m_programGroups );
        return *this;
    }

    unsigned int size() const { return static_cast<unsigned int>( m_programGroups.size() ); }

    operator const OptixProgramGroup*() const { return m_programGroups.data(); }

  private:
    std::vector<OptixProgramGroup> m_programGroups;
};

class Pipeline
{
  public:
    Pipeline() = default;
    Pipeline( OptixDeviceContext                 context,
              const OptixPipelineCompileOptions* pipelineCompileOptions,
              const OptixPipelineLinkOptions*    pipelineLinkOptions,
              const OptixProgramGroup*           programGroups,
              unsigned int                       numProgramGroups )
    {
        OTK_ERROR_CHECK_LOG( optixPipelineCreate( context, pipelineCompileOptions, pipelineLinkOptions, programGroups,
                                                   numProgramGroups, LOG, &LOG_SIZE, &m_pipeline ) );
    }
    Pipeline( const Pipeline& rhs ) = delete;
    Pipeline( Pipeline&& rhs )      = delete;
    ~Pipeline()
    {
        if( m_pipeline )
            optixPipelineDestroy( m_pipeline );
    }
    Pipeline& operator=( const Pipeline& rhs ) = delete;
    Pipeline& operator=( Pipeline&& rhs )
    {
        m_pipeline     = rhs.m_pipeline;
        rhs.m_pipeline = OptixPipeline{};
        return *this;
    }

    operator OptixPipeline() const { return m_pipeline; }

  private:
    OptixPipeline m_pipeline{};
};

void contextLog( uint_t level, const char* tag, const char* text, void* data )
{
    std::ostream* stream = static_cast<std::ostream*>( data );
    std::string   message{ text };
    while( !message.empty() && std::isspace( message.back() ) )
        message.pop_back();
    if( message.empty() )
        return;
    *stream << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << '\n';
}

void setLogger( OptixDeviceContextOptions& options, std::ostream* stream )
{
    options.logCallbackFunction = contextLog;
    options.logCallbackLevel    = 4;
    options.logCallbackData     = stream;
}

template <typename T, unsigned int N>
unsigned int numOf( T ( & )[N] )
{
    return N;
}

class EmptySbtRecord
{
  public:
    EmptySbtRecord() = default;
    EmptySbtRecord( OptixProgramGroup group )
    {
        m_record.resize( 1 );
        m_record.packHeader( 0, group );
        m_record.copyToDevice();
    }
    ~EmptySbtRecord()                                      = default;
    EmptySbtRecord& operator=( const EmptySbtRecord& rhs ) = delete;
    EmptySbtRecord& operator=( EmptySbtRecord&& rhs )      = delete;
    EmptySbtRecord& operator=( OptixProgramGroup group )
    {
        m_record.resize( 1 );
        m_record.packHeader( 0, group );
        m_record.copyToDevice();
        return *this;
    }

    operator CUdeviceptr() { return m_record; }

    unsigned int stride() const { return sizeof( otk::Record<otk::EmptyRecord> ); }

  private:
    otk::SyncRecord<otk::EmptyRecord> m_record;
};

void setDebugIndex( otk::DebugLocation& debug, uint_t x, uint_t y )
{
    debug.enabled       = true;
    debug.debugIndexSet = true;
    debug.debugIndex    = make_uint3( x, y, 0 );
}

class TestDebugLocation : public testing::Test
{
  protected:
    using Predictor = std::function<float3( uint_t x, uint_t y )>;

    void SetUp() override;

    void TearDown() override;

    float3 getResultPixel( uint_t x, uint_t y );
    void   checkResults( const Predictor& predictor );
    uint_t getDumpIndicator();

    OptixDeviceContextOptions   contextOptions{};
    std::ostringstream          log;
    Context                     context;
    OptixModuleCompileOptions   moduleCompileOptions{};
    OptixPipelineCompileOptions pipelineCompileOptions{};
    Module                      module;
    OptixProgramGroupDesc       descs[3];
    OptixProgramGroupOptions    groupOptions{};
    ProgramGroups               groups;
    OptixPipelineLinkOptions    pipelineLinkOptions{};
    Pipeline                    pipeline;
    uint2                       dimensions{ 10, 10 };
    otk::SyncVector<float3>     output;
    otk::SyncVector<uint_t>     dumpIndicator;
    otk::SyncVector<Params>     params;
    EmptySbtRecord              raygen;
    EmptySbtRecord              miss;
    EmptySbtRecord              hitgroup;
    OptixShaderBindingTable     sbt{};
    std::vector<float3>         result;
    bool                        m_resultCopied{};
    float3                      m_miss{ make_float3( -2.0f, -2.0f, -2.0f ) };
    float3                      m_background{ make_float3( -1.0f, -1.0f, -1.0f ) };
    float3                      m_red{ make_float3( 1.0f, 0.0f, 0.0f ) };
    float3                      m_black{ make_float3( 0.0f, 0.0f, 0.0f ) };
    float3                      m_white{ make_float3( 1.0f, 1.0f, 1.0f ) };
};

void TestDebugLocation::SetUp()
{
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    OTK_ERROR_CHECK( optixInit() );
    contextOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    setLogger( contextOptions, &log );
    context                                                 = Context( nullptr, &contextOptions );
    otk::configModuleCompileOptions( moduleCompileOptions );
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "g_params";
    module = Module( context, &moduleCompileOptions, &pipelineCompileOptions, TestDebugLocationCudaText(), TestDebugLocationCudaSize );
    otk::ProgramGroupDescBuilder( descs, module )
        .raygen( "__raygen__debugLocationTest" )
        .miss( "__miss__debugLocationTest" )
        .hitGroupISCH( "__intersection__debugLocationTest", "__closesthit__debugLocationTest" );
    groups   = ProgramGroups( context, descs, numOf( descs ), &groupOptions );
    pipeline = Pipeline( context, &pipelineCompileOptions, &pipelineLinkOptions, groups, groups.size() );
    output.resize( dimensions.x * dimensions.y );
    std::fill( output.begin(), output.end(), m_background );
    output.copyToDevice();
    dumpIndicator.resize(1);
    dumpIndicator[0] = 0U;
    dumpIndicator.copyToDevice();
    params.resize( 1 );
    params[0].width                 = dimensions.x;
    params[0].height                = dimensions.y;
    params[0].image                 = output.typedDevicePtr();
    params[0].miss                  = m_miss;
    params[0].dumpIndicator         = dumpIndicator.typedDevicePtr();
    raygen                          = groups[0];
    miss                            = groups[1];
    hitgroup                        = groups[2];
    sbt.raygenRecord                = raygen;
    sbt.missRecordBase              = miss;
    sbt.missRecordCount             = 1;
    sbt.missRecordStrideInBytes     = miss.stride();
    sbt.hitgroupRecordBase          = hitgroup;
    sbt.hitgroupRecordCount         = 1;
    sbt.hitgroupRecordStrideInBytes = hitgroup.stride();
    result.resize( dimensions.x * dimensions.y );
}

void TestDebugLocation::TearDown()
{
    if( HasFatalFailure() || HasNonfatalFailure() )
    {
        std::cerr << "Log:\n" << log.str();
    }
}

float3 TestDebugLocation::getResultPixel( uint_t x, uint_t y )
{
    if( !m_resultCopied )
    {
        OTK_ERROR_CHECK( cudaMemcpy( result.data(), params[0].image, sizeof( float3 ) * dimensions.x * dimensions.y,
                                     cudaMemcpyDeviceToHost ) );
        m_resultCopied = true;
    }
    return result[dimensions.x * y + x];
}

void TestDebugLocation::checkResults( const Predictor& predictor )
{
    for( uint_t y = 0; y < dimensions.y; ++y )
    {
        for( uint_t x = 0; x < dimensions.x; ++x )
        {
            EXPECT_EQ( predictor( x, y ), getResultPixel( x, y ) ) << "pixel (" << x << ", " << y << ")";
        }
    }
}

uint_t TestDebugLocation::getDumpIndicator()
{
    uint_t indicator{ ~0U };
    OTK_ERROR_CHECK( cudaMemcpy( &indicator, params[0].dumpIndicator, sizeof( uint_t ), cudaMemcpyDeviceToHost ) );
    return indicator;
}

}  // namespace

TEST_F( TestDebugLocation, debugLocationDisabled )
{
    params[0].debug.enabled = false;
    params.copyToDevice();
    OTK_ERROR_CHECK( optixLaunch( pipeline, CUstream{}, params, sizeof( Params ), &sbt, dimensions.x, dimensions.y, 1U ) );

    checkResults( [&]( uint_t, uint_t ) { return m_miss; } );
}

TEST_F( TestDebugLocation, debugLocationEnabledIndexNotSet )
{
    params[0].debug.enabled       = true;
    params[0].debug.debugIndexSet = false;
    params[0].debug.debugIndex    = make_uint3( 5, 5, 0 );  // this value is ignored when debugIndexSet is false
    params.copyToDevice();
    OTK_ERROR_CHECK( optixLaunch( pipeline, CUstream{}, params, sizeof( Params ), &sbt, dimensions.x, dimensions.y, 1U ) );

    checkResults( [&]( uint_t, uint_t ) { return m_miss; } );
}

TEST_F( TestDebugLocation, debugLocationSet )
{
    setDebugIndex( params[0].debug, 5, 5 );
    params.copyToDevice();
    OTK_ERROR_CHECK( optixLaunch( pipeline, CUstream{}, params, sizeof( Params ), &sbt, dimensions.x, dimensions.y, 1U ) );

    checkResults( [&]( uint_t x, uint_t y ) {
        if( x == 5 && y == 5 )
            return m_red;
        if( x >= 3 && x <= 7 && y >= 3 && y <= 7 )
            return m_black;
        if( x >= 1 && x <= 9 && y >= 1 && y <= 9 )
            return m_white;
        return m_miss;
    } );
    EXPECT_EQ( 1U, getDumpIndicator() );
}

TEST_F( TestDebugLocation, dumpSuppressed )
{
    setDebugIndex( params[0].debug, 5, 5 );
    params[0].debug.dumpSuppressed = true;
    params.copyToDevice();
    OTK_ERROR_CHECK( optixLaunch( pipeline, CUstream{}, params, sizeof( Params ), &sbt, dimensions.x, dimensions.y, 1U ) );

    checkResults( [&]( uint_t x, uint_t y ) {
        if( x == 5 && y == 5 )
            return m_red;
        if( x >= 3 && x <= 7 && y >= 3 && y <= 7 )
            return m_black;
        if( x >= 1 && x <= 9 && y >= 1 && y <= 9 )
            return m_white;
        return m_miss;
        } );
    EXPECT_EQ( 0U, getDumpIndicator() );
}

TEST_F( TestDebugLocation, debugLocationSetTopLeftCorner )
{
    setDebugIndex( params[0].debug, 0, 0 );
    params.copyToDevice();
    OTK_ERROR_CHECK( optixLaunch( pipeline, CUstream{}, params, sizeof( Params ), &sbt, dimensions.x, dimensions.y, 1U ) );

    checkResults( [&]( uint_t x, uint_t y ) {
        if( x == 0 && y == 0 )
            return m_red;
        if( x <= 2 && y <= 2 )
            return m_black;
        if( x <= 4 && y <= 4 )
            return m_white;
        return m_miss;
    } );
}

TEST_F( TestDebugLocation, debugLocationSetTopRightCorner )
{
    setDebugIndex( params[0].debug, 9, 0 );
    params.copyToDevice();
    OTK_ERROR_CHECK( optixLaunch( pipeline, CUstream{}, params, sizeof( Params ), &sbt, dimensions.x, dimensions.y, 1U ) );

    checkResults( [&]( uint_t x, uint_t y ) {
        if( x == 9 && y == 0 )
            return m_red;
        if( x >= 7 && y <= 2 )
            return m_black;
        if( x >= 5 && y <= 4 )
            return m_white;
        return m_miss;
    } );
}

TEST_F( TestDebugLocation, debugLocationSetBottomLeftCorner )
{
    setDebugIndex( params[0].debug, 0, 9 );
    params.copyToDevice();
    OTK_ERROR_CHECK( optixLaunch( pipeline, CUstream{}, params, sizeof( Params ), &sbt, dimensions.x, dimensions.y, 1U ) );

    checkResults( [&]( uint_t x, uint_t y ) {
        if( x == 0 && y == 9 )
            return m_red;
        if( x <= 2 && y >= 7 )
            return m_black;
        if( x <= 4 && y >= 5 )
            return m_white;
        return m_miss;
    } );
}

TEST_F( TestDebugLocation, debugLocationSetBottomRightCorner )
{
    setDebugIndex( params[0].debug, 9, 9 );
    params.copyToDevice();
    OTK_ERROR_CHECK( optixLaunch( pipeline, CUstream{}, params, sizeof( Params ), &sbt, dimensions.x, dimensions.y, 1U ) );

    checkResults( [&]( uint_t x, uint_t y ) {
        if( x == 9 && y == 9 )
            return m_red;
        if( x >= 7 && y >= 7 )
            return m_black;
        if( x >= 5 && y >= 5 )
            return m_white;
        return m_miss;
    } );
}
