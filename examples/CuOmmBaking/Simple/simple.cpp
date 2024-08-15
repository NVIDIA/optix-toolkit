// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "SourceDir.h"  // generated from SourceDir.h.in

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>
#include <OptiXToolkit/Util/EXRInputFile.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <OptiXToolkit/CuOmmBaking/CuBuffer.h>

#include <atomic>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <iostream>

// Launches a cuda kernel to convert a cuda texture into a luminance based opacity state texture.
cudaError_t launchBakeLuminanceOpacity(
    uint32_t width,
    uint32_t height,
    uint32_t pitchInBytes,
    float transparencyCutoff,
    float opacityCutoff,
    cudaTextureObject_t texture,
    uint64_t* output );

int g_textureWidth = 2048;
int g_textureHeight = 2048;

const std::vector<uint3> g_indices = {
    {0,1,2},
    {1,3,2}
};

const std::vector<float2> g_texCoords =
{
    { 0.f, 0.f },
    { 1.f, 0.f },
    { 0.f, 1.f },
    { 1.f, 1.f },
};

const std::vector<float3> g_vertices =
{
    { 0.f, 0.f, 0.f },
    { 1.f, 0.f, 0.f },
    { 0.f, 1.f, 0.f },
    { 1.f, 1.f, 0.f },
};

// Check status returned by a CUDA call.
inline void check( cudaError_t status )
{
    if( status != cudaSuccess )
        throw std::runtime_error( cudaGetErrorString( status ) );
}

// Check status returned by a OptiX call.
inline void check( OptixResult status )
{
    if( status != OPTIX_SUCCESS )
        throw std::runtime_error( "OptiX failure." );
}

// Check status returned by a CuOmmBaking call.
inline void check( cuOmmBaking::Result status )
{
    if( status != cuOmmBaking::Result::SUCCESS )
        throw std::runtime_error( "Omm baking failure." );
}

void printUsageAndExit( const char* argv0 )
{
    // clang-format off
    std::cerr
        << "\nUsage  : " << argv0 << " [options]\n"
        << "Options: --help | -h                         Print this usage message\n"
        << "         --texture | -t <filename>           Texture to render (path relative to data folder). Use checkerboard for procedural texture.\n"
        << "         --luminance | -l                    Opacity is based on texture luminance.\n"
        << "\n";
    // clang-format on
    exit( 1 );
}


int main( int argc, char* argv[] )
{
    std::string textureFile = "DuckHole/DuckHole.exr";  // use --texture "" for procedural texture

    bool luminance = false; // alpha is based on luminance
    float transparencyCutoff = 0.05f;
    float opacityCutoff = 0.95f;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        bool              lastArg = ( i == argc - 1 );

        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( ( arg == "--luminance" || arg == "-l" ) )
        {
            luminance = true;
        }
        else if( ( arg == "--texture" || arg == "-t" ) && !lastArg )
        {
            textureFile = argv[++i];
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }
    // Initialize CUDA.
    check( cudaFree( nullptr ) );

    // Initialize OptiX.
    check( optixInit() );

    OptixDeviceContext        context;
    CUcontext                 cuCtx = 0;  // zero means take the current context
    OptixDeviceContextOptions optixOptions = {};
    check( optixDeviceContextCreate( cuCtx, &optixOptions, &context ) );

    // Scoped OptiX context.
    auto contextDestroyer = [&]( struct OptixDeviceContext_t* context ) { try { check( optixDeviceContextDestroy( context ) ); } catch( ... ) {} };
    std::unique_ptr<struct OptixDeviceContext_t, decltype( contextDestroyer )> contextObject( context, contextDestroyer );

    std::string textureFilename( getSourceDir() + "/../Textures/" + textureFile );
    otk::EXRInputFile exrFile;
    exrFile.open( textureFilename );

    // Build a cuda texture.
    cudaChannelFormatDesc desc          = cudaCreateChannelDescHalf4();
    size_t                bytesPerPixel = ( desc.x + desc.y + desc.z + desc.w ) / 8;
    std::vector<char>     data( bytesPerPixel * exrFile.getWidth() * exrFile.getHeight() );
    exrFile.read( data.data(), data.size() );

    size_t bytesPerRow = exrFile.getWidth() * bytesPerPixel;

    CuPitchedBuffer<char> devTexture;
    devTexture.allocAndUpload( bytesPerRow, exrFile.getHeight(), data.data() );

    struct cudaResourceDesc resDesc;
    memset( &resDesc, 0, sizeof( resDesc ) );
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.desc = desc;
    resDesc.res.pitch2D.devPtr = ( void* )devTexture.get();

    resDesc.res.pitch2D.width = exrFile.getWidth();
    resDesc.res.pitch2D.height = exrFile.getHeight();
    resDesc.res.pitch2D.pitchInBytes = devTexture.pitch();

    struct cudaTextureDesc texDesc;
    memset( &texDesc, 0, sizeof( texDesc ) );
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.maxMipmapLevelClamp = 0;
    texDesc.mipmapFilterMode = cudaFilterModePoint;

    cudaTextureObject_t tex;
    check( cudaCreateTextureObject( &tex, &resDesc, &texDesc, NULL ) );

    // Scoped Cuda Texture.
    auto textureDestroyer = [&]( void* tex ) { try { check( cudaDestroyTextureObject( ( cudaTextureObject_t )tex ) ); } catch( ... ) {} };
    std::unique_ptr<void, decltype( textureDestroyer )> textureObject( ( void* )tex, textureDestroyer );

    // Upload geometry data.

    CuBuffer<uint3> d_geoIndices;
    d_geoIndices.allocAndUpload( g_indices );

    CuBuffer<float3> d_geoVertices;
    d_geoVertices.allocAndUpload( g_vertices );

    CuBuffer<float2> d_geoTexCoords;
    d_geoTexCoords.allocAndUpload( g_texCoords );

    // Bake the Opacity Micromap data.

    cuOmmBaking::BakeOptions ommOptions = {};

    cuOmmBaking::TextureDesc texture = {};

    CuBuffer<uint64_t> d_stateBuffer;
    if( luminance )
    {
        // Bake the cuda texture to a luminance based opacity state texture.

        // 2 bits per texel, with a pitch aligned to 64 bits.
        uint32_t pitchInDWords = ( exrFile.getWidth() * 2 + sizeof( uint64_t ) * 8 - 1 ) / ( sizeof( uint64_t ) * 8 );
        uint32_t pitchInBytes = pitchInDWords * sizeof( uint64_t );

        check( d_stateBuffer.alloc( pitchInDWords ) );

        check( launchBakeLuminanceOpacity(
            exrFile.getWidth(), exrFile.getHeight(), pitchInBytes,
            transparencyCutoff, opacityCutoff,
            ( cudaTextureObject_t )textureObject.get(),
            ( uint64_t* )d_stateBuffer.get() ) );

        texture.type = cuOmmBaking::TextureType::STATE;
        texture.state.width = exrFile.getWidth();
        texture.state.height = exrFile.getHeight();
        texture.state.stateBuffer = d_stateBuffer.get();
        // The luminance was point sampled (not pre-filtered). Use a bilinear filter kernel width.
        texture.state.filterKernelWidthInTexels = 1.f;
        texture.state.pitchInBits = pitchInBytes * 8; // pitch in bits, not bytes
    }
    else
    {
        // Use the cuda texture directly.

        texture.type = cuOmmBaking::TextureType::CUDA;
        texture.cuda.texObject = tex;
        texture.cuda.transparencyCutoff = transparencyCutoff;
        texture.cuda.opacityCutoff = opacityCutoff;
    }

    cuOmmBaking::BakeInputDesc input = {};
    input.indexFormat = cuOmmBaking::IndexFormat::I32_UINT;
    input.indexBuffer = d_geoIndices.get();
    input.numIndexTriplets = 2;

    input.texCoordFormat = cuOmmBaking::TexCoordFormat::UV32_FLOAT2;
    input.texCoordBuffer = d_geoTexCoords.get();

    input.numTextures = 1;
    input.textures = &texture;

    // Prepare for baking by query the pre baking info.

    cuOmmBaking::BakeInputBuffers inputBuffer;
    cuOmmBaking::BakeBuffers buffers;
    check( cuOmmBaking::GetPreBakeInfo( &ommOptions, 1, &input, &inputBuffer, &buffers ) );

    // Allocate baking output buffers.

    CuBuffer<>                                   d_ommIndices;
    CuBuffer<>                                   d_ommOutput;
    CuBuffer<OptixOpacityMicromapDesc>           d_ommDescs;
    CuBuffer<OptixOpacityMicromapUsageCount>     d_usageCounts;
    CuBuffer<OptixOpacityMicromapHistogramEntry> d_histogramEntries;
    CuBuffer<>                                   d_temp;

    check( d_ommIndices.alloc( inputBuffer.indexBufferSizeInBytes ) );
    check( d_usageCounts.alloc( inputBuffer.numMicromapUsageCounts ) );
    check( d_ommOutput.alloc( buffers.outputBufferSizeInBytes ) );
    check( d_ommDescs.alloc( buffers.numMicromapDescs ) );
    check( d_histogramEntries.alloc( buffers.numMicromapHistogramEntries ) );
    check( d_temp.alloc( buffers.tempBufferSizeInBytes ) );

    inputBuffer.indexBuffer = d_ommIndices.get();
    inputBuffer.micromapUsageCountsBuffer = d_usageCounts.get();
    buffers.outputBuffer = d_ommOutput.get();
    buffers.perMicromapDescBuffer = d_ommDescs.get();
    buffers.micromapHistogramEntriesBuffer = d_histogramEntries.get();
    buffers.tempBuffer = d_temp.get();

    // Execute the baking.

    check( cuOmmBaking::BakeOpacityMicromaps( &ommOptions, 1, &input, &inputBuffer, &buffers, 0 ) );

    // Download data that is needed on the host to build the OptiX Opacity Micromap Array.

    std::vector<OptixOpacityMicromapHistogramEntry> h_histogram;
    std::vector<OptixOpacityMicromapUsageCount> h_usage;

    h_usage.resize( inputBuffer.numMicromapUsageCounts );
    h_histogram.resize( buffers.numMicromapHistogramEntries );

    d_usageCounts.download( h_usage );
    d_histogramEntries.download( h_histogram );

    // Free buffers that were inputs to the baker, but not needed for OptiX Opacity Micromap Array and GAS builds.

    check( d_stateBuffer.free() );
    check( d_usageCounts.free() );
    check( d_histogramEntries.free() );
    check( d_temp.free() );

    // Build OptiX Opacity Micromap Array.

    OptixMicromapBufferSizes            ommArraySizes = {};
    OptixOpacityMicromapArrayBuildInput ommArrayInput = {};

    ommArrayInput.micromapHistogramEntries = h_histogram.data();
    ommArrayInput.numMicromapHistogramEntries = ( uint32_t )h_histogram.size();
    ommArrayInput.perMicromapDescStrideInBytes = sizeof( OptixOpacityMicromapDesc );
    check( optixOpacityMicromapArrayComputeMemoryUsage( contextObject.get(), &ommArrayInput, &ommArraySizes ) );

    OptixMicromapBuffers ommArrayBuffers = {};
    ommArrayBuffers.outputSizeInBytes = ommArraySizes.outputSizeInBytes;
    ommArrayBuffers.tempSizeInBytes = ommArraySizes.tempSizeInBytes;
    ommArrayInput.perMicromapDescBuffer = buffers.perMicromapDescBuffer;
    ommArrayInput.inputBuffer = buffers.outputBuffer;

    CuBuffer<> d_ommArray;

    check( d_ommArray.alloc( ommArrayBuffers.outputSizeInBytes ) );
    check( d_temp.alloc( ommArrayBuffers.tempSizeInBytes ) );

    ommArrayBuffers.output = d_ommArray.get();
    ommArrayBuffers.temp = d_temp.get();

    check( optixOpacityMicromapArrayBuild( contextObject.get(), 0, &ommArrayInput, &ommArrayBuffers ) );

    // Free the input buffers to the  OptiX Opacity Micromap Array build.

    check( d_ommOutput.free() );
    check( d_ommDescs.free() );
    check( d_temp.free() );

    // Build OptiX GAS.

    OptixBuildInputOpacityMicromap opacityMicromap = {};
    opacityMicromap.indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED;
    opacityMicromap.indexBuffer = inputBuffer.indexBuffer;
    opacityMicromap.indexSizeInBytes =
        ( buffers.indexFormat == cuOmmBaking::IndexFormat::I32_UINT ) ? sizeof( uint32_t ) : sizeof( uint16_t );
    opacityMicromap.micromapUsageCounts = h_usage.data();
    opacityMicromap.numMicromapUsageCounts = h_usage.size();
    opacityMicromap.opacityMicromapArray = ommArrayBuffers.output;

    const unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;

    OptixBuildInputTriangleArray triangleInput = {};
    triangleInput.indexBuffer = d_geoIndices.get();
    triangleInput.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.indexStrideInBytes = 0;
    triangleInput.numIndexTriplets = 2;

    CUdeviceptr vertexBuffers[1] = { d_geoVertices.get() };
    triangleInput.vertexBuffers = vertexBuffers;
    triangleInput.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.vertexStrideInBytes = 0;
    triangleInput.numVertices = 4;

    triangleInput.flags = &flags;
    triangleInput.numSbtRecords = 1;

    triangleInput.opacityMicromap = opacityMicromap;

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray = triangleInput;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes = {};
    check( optixAccelComputeMemoryUsage( contextObject.get(), &accelOptions, &buildInput, 1, &gasBufferSizes ) );

    CuBuffer<> d_gas;
    check( d_gas.alloc( gasBufferSizes.outputSizeInBytes ) );
    check( d_temp.alloc( gasBufferSizes.tempSizeInBytes ) );

    OptixTraversableHandle handle = {};
    check( optixAccelBuild( contextObject.get(), 0, &accelOptions, &buildInput, 1, d_temp.get(), gasBufferSizes.tempSizeInBytes, d_gas.get(), gasBufferSizes.outputSizeInBytes, &handle, nullptr, 0 ) );

    // Free input buffers to the OptiX GAS build.

    check( d_temp.free() );
    check( d_ommIndices.free() );

    ///
    /// The OptiX Opacity Micromap Array and OptiX GAS are ready for use in rendering.
    ///

    // Free the GAS and OptiX Opacity Micromap Array.

    check( d_gas.free() );
    check( d_ommArray.free() );

    return 0;
}
