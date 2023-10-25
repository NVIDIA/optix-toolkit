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

#include <OptiXToolkit/ShaderUtil/DebugLocation.h>

#include "TestDebugLocation.h"

#include "TestDebugLocationParams.h"
#include "TestShaderUtilsPTX.h"

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/Memory/DeviceBuffer.h>
#include <OptiXToolkit/Memory/SyncVector.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <OptiXToolkit/OptiXMemory/SyncRecord.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <gtest/gtest.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda.h>
#include <vector_types.h>

#include <functional>
#include <sstream>
#include <vector>

void PrintTo( const float3& value, std::ostream* stream )
{
    *stream << "float3(" << value.x << ", " << value.y << ", " << value.z << ')';
}

using namespace otk::shaderUtil::testing;

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
            const char*                        PTX,
            size_t                             PTXsize )
    {
        OTK_ERROR_CHECK_LOG2( optixModuleCreateFromPTX( context, moduleCompileOptions, pipelineCompileOptions, PTX,
                                                        PTXsize, LOG, &LOG_SIZE, &m_module ) );
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
        OTK_ERROR_CHECK_LOG2( optixProgramGroupCreate( context, programDescriptions, numProgramGroups, options, LOG,
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
        OTK_ERROR_CHECK_LOG2( optixPipelineCreate( context, pipelineCompileOptions, pipelineLinkOptions, programGroups,
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
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "g_params";
    module = Module( context, &moduleCompileOptions, &pipelineCompileOptions, TestDebugLocation_ptx_text(), TestDebugLocation_ptx_size );
    otk::ProgramGroupDescBuilder( descs, module )
        .raygen( "__raygen__debugLocationTest" )
        .miss( "__miss__debugLocationTest" )
        .hitGroupCHIS( "__closesthit__debugLocationTest", "__intersection__debugLocationTest" );
    groups   = ProgramGroups( context, descs, numOf( descs ), &groupOptions );
    pipeline = Pipeline( context, &pipelineCompileOptions, &pipelineLinkOptions, groups, groups.size() );
    output.resize( dimensions.x * dimensions.y );
    std::fill( output.begin(), output.end(), m_background );
    output.copyToDevice();
    params.resize( 1 );
    params[0].width                 = dimensions.x;
    params[0].height                = dimensions.y;
    params[0].image                 = output.typedDevicePtr();
    params[0].miss                  = m_miss;
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
