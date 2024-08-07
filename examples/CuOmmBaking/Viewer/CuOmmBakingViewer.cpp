
//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "CuOmmBakingApp.h"
#include "CuOmmBakingViewerKernelCuda.h"
#include "LaunchParams.h"

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sstream>

using namespace ommBakingApp;

__host__ void launchEvaluateOmmOpacity(
    uint32_t width,
    uint32_t height,
    uint32_t pitchInBytes,
    float transpacentyCutoff,
    float opacityCutoff,
    uint64_t* output );

void bakingCheck( cuOmmBaking::Result res, const char* call, const char* file, unsigned int line )
{
    if( res != cuOmmBaking::Result::SUCCESS )
    {
        std::stringstream ss;
        ss << "CuOmmBaking call '" << call << "' failed: " << file << ':' << line << ")\n";
        throw std::runtime_error( ss.str().c_str() );
    }
}

#define BAKING_CHECK( call )                         \
    ::bakingCheck( call, #call, __FILE__, __LINE__ )

//------------------------------------------------------------------------------
// OmmBakingViewer
// Shows basic use of OptiX Opacity Micromap baking.
//------------------------------------------------------------------------------

// Shader binding table records
template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

// Per device scene data
struct PerDeviceState
{
    CuBuffer<uint3>             d_geoIndices   = {};
    CuBuffer<float2>            d_texCoords = {};

    CuBuffer<>                  d_gas      = {}; // device side buffer for d_gas
    CuBuffer<>                  d_ommArray = {}; // device side buffer for omm array

    CuBuffer<Params>            d_params         = {};  // Device-side copy of params

    CuBuffer<>                  d_texture        = {};

    CuBuffer<RayGenSbtRecord>   d_rayGenSbtRecord    = {};
    CuBuffer<MissSbtRecord>     d_missSbtRecord      = {};
    CuBuffer<HitGroupSbtRecord> d_hitGroupSbtRecords = {};

    OptixTraversableHandle      gas_handle = 0;   // Traversable handle for geometry acceleration structure (d_gas)

    CudaTexture                 texture;

    size_t                      texture_width = 0;
    size_t                      texture_height = 0;
    size_t                      texture_pitch_in_bytes = 0;

    OptixShaderBindingTable     sbt = {};  // Shader binding table

    Params                      params = {};  // Host-side copy of parameters for OptiX launch
};

// App takes care of boiler plate optix setup and GUI handling
class OmmBakingViewer : public OmmBakingApp
{
  public:
      OmmBakingViewer( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
        : OmmBakingApp( appTitle, width, height, outFileName, glInterop )
    {
    }

    void initOptixPipelines( const char* moduleCode, const size_t moduleCodeSize );
    void setTextureName( const char* textureName ) { m_textureName = textureName; }

  protected:

    void keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods ) override;

    void createSBT( const PerDeviceOptixState& state ) override;
    void buildAccel( const PerDeviceOptixState& state ) override;
    void performLaunch( const PerDeviceOptixState& state, uchar4* result_buffer ) override;

  private:

    void createTexture( int device_idx );

    bool   m_visualizeUnknowns = false;
    float3 m_backgroundColor = float3{ 0.1f, 0.1f, 0.5f };

    std::vector<PerDeviceState> m_state;

    std::string m_textureName;
};

void OmmBakingViewer::createTexture( int device_idx )
{
    PerDeviceState &state = m_state[device_idx];

    if( m_textureName.empty() )
    {
        // Bake the procedural opacity state texture input

        float transpacentyCutoff = 0.f;
        float opacityCutoff = 1.f;

        size_t width = 512;
        size_t height = 512;
        size_t pitchInBytes = 8 * ( ( 2 * width + sizeof(uint64_t) * 8 - 1 ) / ( sizeof( uint64_t ) * 8 ) );

        CuBuffer<> d_texture = {};

        d_texture.alloc( height * pitchInBytes );

        launchEvaluateOmmOpacity(
            (uint32_t)width,
            (uint32_t)height,
            (uint32_t)pitchInBytes,
            transpacentyCutoff,
            opacityCutoff,
            ( uint64_t* )d_texture.get() );

        // commit texture to device state

        std::swap( state.d_texture, d_texture );
        std::swap( state.texture_width, width );
        std::swap( state.texture_height, height );
        std::swap( state.texture_pitch_in_bytes, pitchInBytes );
    }
    else
    {
        // Load texture

        CudaTexture texture( m_textureName );

        // commit texture to device state

        std::swap( state.texture, texture );
    }
}

//------------------------------------------------------------------------------
// Scene build
//------------------------------------------------------------------------------

void OmmBakingViewer::buildAccel( const PerDeviceOptixState& optixState )
{
    PerDeviceState& state = m_state[optixState.device_idx];

    std::vector<uint3> g_indices = {
        {0,1,2},
        {1,3,2}
    };

    std::vector<float2> g_texCoords =
    {
        {0.f, 0.f},
        {1.f, 0.f},
        {0.f, 1.f},
        {1.f, 1.f},
    };

    std::vector<float3> g_vertices =
    {
        {0.f, 0.f, 0.f},
        {1.f, 0.f, 0.f},
        {0.f, 1.f, 0.f},
        {1.f, 1.f, 0.f},
    };

    // Upload geometry data

    CuBuffer<uint3> d_geoIndices;
    OTK_ERROR_CHECK( d_geoIndices.allocAndUpload( g_indices ) );

    CuBuffer<float3> d_vertices;
    OTK_ERROR_CHECK( d_vertices.allocAndUpload( g_vertices ) );

    CuBuffer<float2> d_texCoords;
    OTK_ERROR_CHECK( d_texCoords.allocAndUpload( g_texCoords ) );

    // Bake the Opacity Micromap data

    cuOmmBaking::BakeOptions ommOptions = {};
    ommOptions.flags = cuOmmBaking::BakeFlags::ENABLE_POST_BAKE_INFO;

    cuOmmBaking::TextureDesc texture = {};

    if( state.texture.get() )
    {
        texture.type = cuOmmBaking::TextureType::CUDA;
        texture.cuda.texObject = state.texture.get();
        texture.cuda.transparencyCutoff = 0.f;
        texture.cuda.opacityCutoff = 1.f;
    }
    else
    {
        texture.type = cuOmmBaking::TextureType::STATE;
        texture.state.stateBuffer = state.d_texture.get();
        texture.state.width = state.texture_width;
        texture.state.height = state.texture_height;
        texture.state.pitchInBits = state.texture_pitch_in_bytes * 8;
        texture.state.addressMode[0] = cudaAddressModeMirror;
        texture.state.addressMode[1] = cudaAddressModeMirror;
        // the procedural is pre-filtered.
        texture.state.filterKernelWidthInTexels = 0.f;
    }

    cuOmmBaking::BakeInputDesc input = {};
    input.indexFormat = cuOmmBaking::IndexFormat::I32_UINT;
    input.indexBuffer = d_geoIndices.get();
    input.numIndexTriplets = 2;

    input.texCoordFormat = cuOmmBaking::TexCoordFormat::UV32_FLOAT2;
    input.texCoordBuffer = d_texCoords.get();

    input.numTextures = 1;
    input.textures = &texture;

    // Prepare for baking by querying the pre baking info

    cuOmmBaking::BakeInputBuffers inputBuffers = {};
    cuOmmBaking::BakeBuffers buffers = {};
    BAKING_CHECK( cuOmmBaking::GetPreBakeInfo( &ommOptions, 1, &input, &inputBuffers, &buffers ) );

    CuBuffer<>                                  d_ommIndex;
    CuBuffer<OptixOpacityMicromapUsageCount>    d_usageCounts;
    std::vector<OptixOpacityMicromapUsageCount> h_usageCounts;
    
    CuBuffer<> d_ommArray;

    // Allocate buffers for bake input
    d_ommIndex.alloc( inputBuffers.indexBufferSizeInBytes );
    d_usageCounts.alloc( inputBuffers.numMicromapUsageCounts );
    
    {
        CuBuffer<>                         d_ommOutput;
        CuBuffer<OptixOpacityMicromapDesc> d_ommDesc;

        // Allocate buffers for opacity array build input
        d_ommOutput.alloc( buffers.outputBufferSizeInBytes );
        d_ommDesc.alloc( buffers.numMicromapDescs );

        cuOmmBaking::PostBakeInfo h_postBuildInfo = {};
        std::vector<OptixOpacityMicromapHistogramEntry> h_histogramEntries;
        
        // Execute the baking
        {
            CuBuffer<OptixOpacityMicromapHistogramEntry> d_histogramEntries;
            CuBuffer<cuOmmBaking::PostBakeInfo>          d_postBakeInfo;
            CuBuffer<>                                   d_temp;

            // Allocate d_temp buffer and buffers to be downloaded after baking.
            d_histogramEntries.alloc( buffers.numMicromapHistogramEntries );
            d_postBakeInfo.alloc( 1 );
            d_temp.alloc( buffers.tempBufferSizeInBytes );

            inputBuffers.indexBuffer = d_ommIndex.get();
            inputBuffers.micromapUsageCountsBuffer = d_usageCounts.get();

            buffers.outputBuffer = d_ommOutput.get();
            buffers.perMicromapDescBuffer = d_ommDesc.get();
            buffers.micromapHistogramEntriesBuffer = d_histogramEntries.get();
            buffers.postBakeInfoBuffer = d_postBakeInfo.get();
            buffers.tempBuffer = d_temp.get();

            BAKING_CHECK( cuOmmBaking::BakeOpacityMicromaps( &ommOptions, 1, &input, &inputBuffers, &buffers, 0 ) );

            // Download data that is needed on the host to build the OptiX Opacity Micromap Array
            h_usageCounts.resize( inputBuffers.numMicromapUsageCounts );
            h_histogramEntries.resize( buffers.numMicromapHistogramEntries );

            OTK_ERROR_CHECK( d_postBakeInfo.download( &h_postBuildInfo ) );
            OTK_ERROR_CHECK( d_histogramEntries.download( h_histogramEntries ) );
            OTK_ERROR_CHECK( d_usageCounts.download( h_usageCounts ) );

        }

        // Build OptiX Opacity Micromap Array

        OptixMicromapBuffers ommArrayBuffers = {};
        if( h_postBuildInfo.numMicromapDescs )
        {
            OptixMicromapBufferSizes            ommArraySizes = {};
            OptixOpacityMicromapArrayBuildInput ommArrayInput = {};

            ommArrayInput.micromapHistogramEntries = h_histogramEntries.data();
            ommArrayInput.numMicromapHistogramEntries = ( uint32_t )h_histogramEntries.size();
            ommArrayInput.perMicromapDescStrideInBytes = sizeof( OptixOpacityMicromapDesc );
            OTK_ERROR_CHECK( optixOpacityMicromapArrayComputeMemoryUsage( optixState.context, &ommArrayInput, &ommArraySizes ) );

            CuBuffer<> temp;
            OTK_ERROR_CHECK( temp.alloc( ommArraySizes.tempSizeInBytes ) );
            OTK_ERROR_CHECK( d_ommArray.alloc( ommArraySizes.outputSizeInBytes ) );

            ommArrayBuffers.outputSizeInBytes = d_ommArray.byteSize();
            ommArrayBuffers.output = d_ommArray.get();
            ommArrayBuffers.tempSizeInBytes = temp.byteSize();
            ommArrayBuffers.temp = temp.get();

            ommArrayInput.perMicromapDescBuffer = buffers.perMicromapDescBuffer;
            ommArrayInput.inputBuffer = buffers.outputBuffer;

            OTK_ERROR_CHECK( optixOpacityMicromapArrayBuild( optixState.context, 0, &ommArrayInput, &ommArrayBuffers ) );
        }
    }

    // Build OptiX GAS

    const unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;

    OptixBuildInputTriangleArray triangleInput = {};
    triangleInput.indexBuffer = d_geoIndices.get();
    triangleInput.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.indexStrideInBytes = 0;
    triangleInput.numIndexTriplets = 2;

    CUdeviceptr vertexBuffers[1] = { d_vertices.get() };
    triangleInput.vertexBuffers = vertexBuffers;
    triangleInput.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.vertexStrideInBytes = 0;
    triangleInput.numVertices = 4;

    triangleInput.flags = &flags;
    triangleInput.numSbtRecords = 1;

    triangleInput.opacityMicromap.indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED;
    triangleInput.opacityMicromap.indexBuffer = inputBuffers.indexBuffer;
    triangleInput.opacityMicromap.indexSizeInBytes =
        ( buffers.indexFormat == cuOmmBaking::IndexFormat::I32_UINT ) ? sizeof( uint32_t ) : sizeof( uint16_t );
    triangleInput.opacityMicromap.micromapUsageCounts = h_usageCounts.data();
    triangleInput.opacityMicromap.numMicromapUsageCounts = h_usageCounts.size();
    triangleInput.opacityMicromap.opacityMicromapArray = d_ommArray.get();

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray = triangleInput;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes = {};
    OTK_ERROR_CHECK( optixAccelComputeMemoryUsage( optixState.context, &accelOptions, &buildInput, 1, &gasBufferSizes ) );

    CuBuffer<> d_temp;
    CuBuffer<> d_gas;

    size_t compactedSizeOffset = ( gasBufferSizes.tempSizeInBytes + sizeof( size_t ) - 1 ) & ( ~( sizeof( size_t ) - 1 ) );

    OTK_ERROR_CHECK( d_gas.alloc( gasBufferSizes.outputSizeInBytes ) );
    OTK_ERROR_CHECK( d_temp.alloc( compactedSizeOffset + sizeof( size_t ) ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = d_temp.get(compactedSizeOffset);

    OptixTraversableHandle handle = {};
    OTK_ERROR_CHECK( optixAccelBuild( optixState.context, 0, &accelOptions, &buildInput, 1, d_temp.get(), d_temp.byteSize(), d_gas.get(), d_gas.byteSize(), &handle, &emitProperty, 1));

    size_t compacted_gas_size;
    OTK_ERROR_CHECK( cudaMemcpy( &compacted_gas_size, ( void* )emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

    // Free input buffers to the GAS build
    d_temp.free();
    d_ommIndex.free();

    // Compact GAS

    if( compacted_gas_size < gasBufferSizes.outputSizeInBytes )
    {
        CuBuffer<> d_compactedGas;
        OTK_ERROR_CHECK( d_compactedGas.alloc( compacted_gas_size ) );

        // use handle as input and output
        OTK_ERROR_CHECK( optixAccelCompact( optixState.context, 0, handle, d_compactedGas.get(),
            d_compactedGas.byteSize(), &handle));

        std::swap( d_gas, d_compactedGas );
    }

    // Commit scene to device state

    std::swap( state.d_ommArray, d_ommArray );
    std::swap( state.d_gas, d_gas );
    std::swap( state.gas_handle, handle );
    std::swap( state.d_geoIndices, d_geoIndices );
    std::swap( state.d_texCoords ,d_texCoords );
}

//------------------------------------------------------------------------------
// OptiX setup
//------------------------------------------------------------------------------

void OmmBakingViewer::initOptixPipelines( const char* moduleCode, const size_t moduleCodeSize )
{
    OTK_ERROR_CHECK( optixInit() );

    int numDevices;
    OTK_ERROR_CHECK( cudaGetDeviceCount( &numDevices ) );
    m_state.resize( numDevices );

    for( int i = 0; i < numDevices; ++i )
    {
        OTK_ERROR_CHECK( cudaSetDevice( i ) );
        OTK_ERROR_CHECK( cudaFree( 0 ) );
        createTexture( i );
    }

    OmmBakingApp::initOptixPipelines( moduleCode, moduleCodeSize, numDevices );
}

void OmmBakingViewer::createSBT( const PerDeviceOptixState& optixState )
{
    PerDeviceState& state = m_state[optixState.device_idx];

    CuBuffer<RayGenSbtRecord>   d_rayGenSbtRecord;
    CuBuffer<MissSbtRecord>     d_missSbtRecord;
    CuBuffer<HitGroupSbtRecord> d_hitGroupSbtRecords;

    RayGenSbtRecord rg_sbt = {};
    OTK_ERROR_CHECK( optixSbtRecordPackHeader( optixState.raygen_prog_group, &rg_sbt ) );
    d_rayGenSbtRecord.allocAndUpload( 1, &rg_sbt );

    MissSbtRecord ms_sbt;
    ms_sbt.data.background_color = m_backgroundColor;
    OTK_ERROR_CHECK( optixSbtRecordPackHeader( optixState.miss_prog_group, &ms_sbt ) );
    d_missSbtRecord.allocAndUpload( 1, &ms_sbt );

    HitGroupSbtRecord hg_sbt;
    hg_sbt.data.texture_id = state.texture.get();
    hg_sbt.data.indices = ( const uint3* )state.d_geoIndices.get();
    hg_sbt.data.texCoords = ( const float2* )state.d_texCoords.get();
    OTK_ERROR_CHECK( optixSbtRecordPackHeader( optixState.hitgroup_prog_group, &hg_sbt ) );
    d_hitGroupSbtRecords.allocAndUpload( 1, &hg_sbt );

    std::swap( state.d_rayGenSbtRecord, d_rayGenSbtRecord );
    std::swap( state.d_missSbtRecord, d_missSbtRecord );
    std::swap( state.d_hitGroupSbtRecords, d_hitGroupSbtRecords );

    state.sbt.raygenRecord = state.d_rayGenSbtRecord.get();
    state.sbt.missRecordBase = state.d_missSbtRecord.get();
    state.sbt.missRecordStrideInBytes = sizeof( MissSbtRecord );
    state.sbt.missRecordCount = state.d_missSbtRecord.count();
    state.sbt.hitgroupRecordBase = state.d_hitGroupSbtRecords.get();
    state.sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
    state.sbt.hitgroupRecordCount = state.d_hitGroupSbtRecords.count();
}

//------------------------------------------------------------------------------
// OptiX launches
//------------------------------------------------------------------------------

void OmmBakingViewer::performLaunch( const PerDeviceOptixState& optixState, uchar4* result_buffer )
{
    PerDeviceState& state = m_state[optixState.device_idx];

    const unsigned int numDevices = static_cast< unsigned int >( m_state.size() );

    state.params.result_buffer = result_buffer;
    state.params.image_width = getWidth();
    state.params.image_height = getHeight();
    state.params.traversable_handle = state.gas_handle;
    state.params.device_idx = optixState.device_idx;
    state.params.num_devices = numDevices;
    state.params.eye = getEye();
    state.params.view_dims = getViewDims();
    state.params.visualize_omm = m_visualizeUnknowns;

    // Make sure a device-side copy of the params has been allocated
    OTK_ERROR_CHECK( state.d_params.allocIfRequired( 1 ) );
    OTK_ERROR_CHECK( state.d_params.upload( &state.params ) );

    // Peform the OptiX launch, with each device doing a part of the work
    unsigned int launchHeight = ( state.params.image_height + numDevices - 1 ) / numDevices;
    OTK_ERROR_CHECK( optixLaunch( optixState.pipeline,  // OptiX pipeline
        optixState.stream,           // Stream for launch
        state.d_params.get(),       // Launch params
        sizeof( Params ),           // Param size in bytes
        &state.sbt,                 // Shader binding table
        state.params.image_width,   // Launch width
        launchHeight,               // Launch height
        1                           // launch depth
    ) );
}

//------------------------------------------------------------------------------
// User Interaction via GLFW
//------------------------------------------------------------------------------

void OmmBakingViewer::keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods )
{
    OmmBakingApp::keyCallback( window, key, scancode, action, mods );

    if( action != GLFW_PRESS )
        return;

    if( key == GLFW_KEY_K )
        m_visualizeUnknowns = !m_visualizeUnknowns;
}


//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------

void printUsage( const char* argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n\n";
    std::cout << "Options:  --texture <texturefile.exr>, --dim=<width>x<height>, --file <outputfile.ppm> --no-gl-interop\n";
    std::cout << "Keyboard: <ESC>:exit, WASD:pan, QE:zoom, C:recenter, K:visualize unknowns\n";
    std::cout << "Mouse:    <LMB>:pan, <RMB>:zoom\n" << std::endl;
    exit(0);
}

int main( int argc, char* argv[] )
{
    int         windowWidth  = 768;
    int         windowHeight = 768;
    const char* textureName  = "";
    const char* outFileName  = "";
    bool        glInterop    = true;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        const bool        lastArg = ( i == argc - 1 );

        if( ( arg == "--texture" ) && !lastArg )
            textureName = argv[++i];
        else if( ( arg == "--file" ) && !lastArg )
            outFileName = argv[++i];
        else if( arg.substr( 0, 6 ) == "--dim=" )
            otk::parseDimensions( arg.substr( 6 ).c_str(), windowWidth, windowHeight );
        else if( arg == "--no-gl-interop" )
            glInterop = false;
        else
            printUsage( argv[0] );
    }

    OmmBakingViewer app( "Opacity Micromap Viewer", windowWidth, windowHeight, outFileName, glInterop );
    app.setTextureName( textureName );
    app.initOptixPipelines( CuOmmBakingViewerCudaText(), CuOmmBakingViewerCudaSize );
    app.startLaunchLoop();
    
    return 0;
}
