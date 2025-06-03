// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "TestTextureFootprint.h"

#include "TextureFootprintCuda.h"

#include <OptiXToolkit/DemandLoading/Texture2DFootprint.h>
#include <OptiXToolkit/DemandLoading/TextureSampler.h>
#include <OptiXToolkit/DemandLoading/TileIndexing.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/OptiXMemory/CompileOptions.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <optix_types.h>

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <ostream>

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<int>        MissSbtRecord;
typedef SbtRecord<int>        HitGroupSbtRecord;

namespace {  // anonymous

//------------------------------------------------------------------------------
// Helper functions to interpret texture footprints

struct TileCoords
{
    unsigned int              mipLevel;
    unsigned int              numCoords;
    static const unsigned int MAX_TILE_COORDS = 4;
    uint2                     coords[MAX_TILE_COORDS];
};

inline bool isSuperset( const TileCoords& a, const TileCoords& b )
{
    for( unsigned int j = 0; j < b.numCoords; ++j )
    {
        bool found = false;
        for( unsigned int i = 0; i < a.numCoords; ++i )
        {
            if( 0 == memcmp( &a.coords[i], &b.coords[j], sizeof( uint2 ) ) )
            {
                found = true;
                break;
            }
        }
        if( !found )
            return false;
    }
    return true;
}

std::ostream& operator<<( std::ostream& out, const TileCoords& coords )
{
    out << "{ " << coords.mipLevel << ", " << coords.numCoords << ", {";
    for( unsigned int i = 0; i < coords.numCoords; ++i )
    {
        if( i > 0 )
            out << ", ";
        out << "{" << coords.coords[i].x << ", " << coords.coords[i].y << "}";
    }
    out << "}}";
    return out;
}

inline int indexOfFirstSetBit( uint64_t m )
{
    for( int i = 0; i < 64; ++i )
    {
        if( ( ( m >> i ) & 1 ) != 0 )
            return i;
    }
    return 64;
}

TileCoords getTileCoordsFromFootprint( const demandLoading::Texture2DFootprint& footprint )
{
    TileCoords result{};
    result.mipLevel = footprint.level;

    uint64_t     mask      = footprint.mask;
    unsigned int numCoords = 0;
    while( mask != 0 )
    {
        int bitNum = indexOfFirstSetBit( mask );
        mask ^= ( 1ull << bitNum );

        int offsetx = bitNum % 8;
        if( offsetx + footprint.dx >= 8 )
            offsetx -= 8;
        result.coords[numCoords].x = footprint.tileX * 8 + offsetx;

        int offsety = bitNum / 8;
        if( offsety + footprint.dy >= 8 )
            offsety -= 8;
        result.coords[numCoords].y = footprint.tileY * 8 + offsety;

        numCoords++;
    }
    result.numCoords = numCoords;
    return result;
}

inline __host__ __device__ int wrapTileCoord( int x, int xmax )
{
    x = ( x >= 0 ) ? x : x + xmax;
    return ( x < xmax ) ? x : x - xmax;
}

inline void wrapTileCoords( TileCoords& tileCoords, unsigned int levelWidthInTiles, unsigned int levelHeightInTiles )
{
    for( unsigned int i = 0; i < tileCoords.numCoords; ++i )
    {
        tileCoords.coords[i].x = wrapTileCoord( tileCoords.coords[i].x, levelWidthInTiles );
        tileCoords.coords[i].y = wrapTileCoord( tileCoords.coords[i].y, levelHeightInTiles );
    }
}

//------------------------------------------------------------------------------
// Test Fixture

const int DEFAULT_TEXTURE_SIZE = 1024;

class TextureFootprintFixture
{
  private:
    const std::string    m_raygenProgram = "TestTextureFootprint.ptx";
    cudaTextureObject_t  m_texture{};
    cudaMipmappedArray_t m_mipmapArray{};

    unsigned int m_granularity       = 0;
    unsigned int m_tileWidth         = 64;
    unsigned int m_tileHeight        = 64;
    unsigned int m_textureWidth      = DEFAULT_TEXTURE_SIZE;
    unsigned int m_textureHeight     = DEFAULT_TEXTURE_SIZE;
    unsigned int m_mipTailFirstLevel = 4;

  public:
    void setTextureWidth( unsigned int w ) { m_textureWidth = w; }
    void setTextureHeight( unsigned int h ) { m_textureHeight = h; }

  private:
    void initTexture( int width, int height, cudaTextureAddressMode addressMode )
    {
        ASSERT_TRUE( static_cast<unsigned int>( width ) <= MAX_TEXTURE_DIM && static_cast<unsigned int>( height ) <= MAX_TEXTURE_DIM );

        // Allocate mipmapped array
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 32, 32, 32, cudaChannelFormatKindFloat );
        cudaExtent            extent      = make_cudaExtent( width, height, 0 );
        int numLevels = static_cast<int>( 1.f + std::log2( static_cast<float>( std::max( width, height ) ) ) );
        OTK_ERROR_CHECK( cudaMallocMipmappedArray( &m_mipmapArray, &channelDesc, extent, numLevels ) );

        // Fill in each miplevel.
        int levelWidth  = width;
        int levelHeight = height;
        for( int level = 0; level < numLevels; ++level )
        {
            // Get the array for this miplevel.
            cudaArray_t miplevelArray;
            OTK_ERROR_CHECK( cudaGetMipmappedArrayLevel( &miplevelArray, m_mipmapArray, level ) );

            // Sanity check the dimensions of the array for this miplevel.
            cudaChannelFormatDesc levelDesc;
            cudaExtent            levelExtent;
            unsigned int          levelFlags;
            OTK_ERROR_CHECK( cudaArrayGetInfo( &levelDesc, &levelExtent, &levelFlags, miplevelArray ) );
            ASSERT_EQ( static_cast<size_t>( levelWidth ), levelExtent.width );
            ASSERT_EQ( static_cast<size_t>( levelHeight ), levelExtent.height );

            // Copy texel values into the miplevel array.
            std::vector<float4> texels( width * height, make_float4( 0.f, 0.f, 0.f, 0.f ) );
            size_t              levelWidthInBytes = levelWidth * sizeof( float4 );
            size_t              pitch             = levelWidthInBytes;
            OTK_ERROR_CHECK( cudaMemcpy2DToArray( miplevelArray, 0, 0, texels.data(), pitch, levelWidthInBytes,
                                                    levelHeight, cudaMemcpyHostToDevice ) );

            levelWidth  = ( levelWidth ) / 2;
            levelHeight = ( levelHeight ) / 2;
        }

        // Create resource description
        cudaResourceDesc resDesc{};
        resDesc.resType           = cudaResourceTypeMipmappedArray;
        resDesc.res.mipmap.mipmap = m_mipmapArray;

        // Construct texture description with various options.
        cudaTextureDesc texDesc{};
        texDesc.addressMode[0]               = addressMode;
        texDesc.addressMode[1]               = addressMode;
        texDesc.filterMode                   = cudaFilterModeLinear;
        texDesc.maxMipmapLevelClamp          = float( numLevels - 1 );
        texDesc.minMipmapLevelClamp          = 0.f;
        texDesc.mipmapFilterMode             = cudaFilterModeLinear;
        texDesc.normalizedCoords             = 1;
        texDesc.readMode                     = cudaReadModeElementType;
        texDesc.maxAnisotropy                = 16;
        texDesc.disableTrilinearOptimization = CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION;

        // Create texture object
        OTK_ERROR_CHECK( cudaCreateTextureObject( &m_texture, &resDesc, &texDesc, nullptr /*cudaResourceViewDesc*/ ) );
    }

    void destroyTexture()
    {
        OTK_ERROR_CHECK( cudaDestroyTextureObject( m_texture ) );
        OTK_ERROR_CHECK( cudaFreeMipmappedArray( m_mipmapArray ) );
    }

    static unsigned int getGranularityForTileSize( unsigned int tileWidth, unsigned int tileHeight )
    {
        if( tileWidth == 64 && tileHeight == 64 )
            return 11;
        else if( tileWidth == 128 && tileHeight == 64 )
            return 12;
        else if( tileWidth == 128 && tileHeight == 128 )
            return 13;
        else
            return 1;
    }

    static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
    {
        std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
    }

  public:
    int runTest( const char*                         entryFunction,
                 const std::vector<FootprintInputs>& inputs,
                 const std::vector<TileCoords>&      expectedTileCoords,
                 cudaTextureAddressMode              addressMode )
    {
        try
        {
            char   log[2048];  // For error reporting from OptiX creation functions
            size_t sizeof_log = sizeof( log );

            //
            // Initialize CUDA and create OptiX context
            //
            OptixDeviceContext context = nullptr;
            CUdevice           device;
            {
                // Initialize CUDA
                OTK_ERROR_CHECK( cudaFree( nullptr ) );

                // Ignore tests for devices that do not support texture footprints
                OTK_ERROR_CHECK( cuCtxGetDevice( &device ) );
                int sparseSupport = 0;
                OTK_ERROR_CHECK( cuDeviceGetAttribute( &sparseSupport, CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED, device ) );
                if( !sparseSupport )
                    return 0;

                CUcontext cuCtx = 0;  // zero means take the current context
                OTK_ERROR_CHECK( optixInit() );
                OptixDeviceContextOptions options = {};
                options.logCallbackFunction       = &context_log_cb;
                options.logCallbackLevel          = 4;
                OTK_ERROR_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
            }

            //
            // Create module
            //
            OptixModule                 module                   = nullptr;
            OptixPipelineCompileOptions pipeline_compile_options = {};
            {
                OptixModuleCompileOptions module_compile_options = {};
                otk::configModuleCompileOptions( module_compile_options );
                module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

                pipeline_compile_options.usesMotionBlur        = false;
                pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
                pipeline_compile_options.numPayloadValues      = 2;
                pipeline_compile_options.numAttributeValues = 2;
                pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
                pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

                OTK_ERROR_CHECK( optixModuleCreate( context, &module_compile_options, &pipeline_compile_options,
                                                TestTextureFootprintCudaText(), TestTextureFootprintCudaSize, log,
                                                &sizeof_log, &module ) );
            }

            //
            // Create program groups, including NULL miss and hitgroups
            //
            OptixProgramGroup raygen_prog_group   = nullptr;
            OptixProgramGroup miss_prog_group     = nullptr;
            OptixProgramGroup hitgroup_prog_group = nullptr;
            {
                OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

                OptixProgramGroupDesc raygen_prog_group_desc    = {};  //
                raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                raygen_prog_group_desc.raygen.module            = module;
                raygen_prog_group_desc.raygen.entryFunctionName = entryFunction;
                OTK_ERROR_CHECK( optixProgramGroupCreate( context, &raygen_prog_group_desc,
                                                      1,  // num program groups
                                                      &program_group_options, log, &sizeof_log, &raygen_prog_group ) );

                // Leave miss group's module and entryfunc name null
                OptixProgramGroupDesc miss_prog_group_desc = {};
                miss_prog_group_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_MISS;
                OTK_ERROR_CHECK( optixProgramGroupCreate( context, &miss_prog_group_desc,
                                                      1,  // num program groups
                                                      &program_group_options, log, &sizeof_log, &miss_prog_group ) );

                // Leave hit group's module and entryfunc name null
                OptixProgramGroupDesc hitgroup_prog_group_desc = {};
                hitgroup_prog_group_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                OTK_ERROR_CHECK( optixProgramGroupCreate( context, &hitgroup_prog_group_desc,
                                                      1,  // num program groups
                                                      &program_group_options, log, &sizeof_log, &hitgroup_prog_group ) );
            }

            //
            // Link pipeline
            //
            OptixPipeline pipeline = nullptr;
            {
                const uint32_t    max_trace_depth  = 0;
                OptixProgramGroup program_groups[] = {raygen_prog_group};

                OptixPipelineLinkOptions pipeline_link_options = {};
                pipeline_link_options.maxTraceDepth            = max_trace_depth;
                OTK_ERROR_CHECK( optixPipelineCreate( context, &pipeline_compile_options, &pipeline_link_options,
                                                  program_groups, sizeof( program_groups ) / sizeof( program_groups[0] ),
                                                  log, &sizeof_log, &pipeline ) );

                OptixStackSizes stack_sizes = {};
                for( auto& prog_group : program_groups )
                {
#if OPTIX_VERSION < 70700
                    OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
#else
                    OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, pipeline ) );
#endif
                }

                uint32_t direct_callable_stack_size_from_traversal;
                uint32_t direct_callable_stack_size_from_state;
                uint32_t continuation_stack_size;
                OTK_ERROR_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                         0,  // maxCCDepth
                                                         0,  // maxDCDEpth
                                                         &direct_callable_stack_size_from_traversal,
                                                         &direct_callable_stack_size_from_state, &continuation_stack_size ) );
                OTK_ERROR_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                                        direct_callable_stack_size_from_state, continuation_stack_size,
                                                        2  // maxTraversableDepth
                                                        ) );
            }

            //
            // Set up shader binding table
            //
            OptixShaderBindingTable sbt = {};
            {
                CUdeviceptr  raygen_record;
                const size_t raygen_record_size = sizeof( RayGenSbtRecord );
                OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &raygen_record ), raygen_record_size ) );
                RayGenSbtRecord rg_sbt;
                OTK_ERROR_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
                rg_sbt.data = {0.462f, 0.725f, 0.f};
                OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( raygen_record ), &rg_sbt, raygen_record_size,
                                               cudaMemcpyHostToDevice ) );

                CUdeviceptr miss_record;
                size_t      miss_record_size = sizeof( MissSbtRecord );
                OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &miss_record ), miss_record_size ) );
                RayGenSbtRecord ms_sbt;
                OTK_ERROR_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
                OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( miss_record ), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice ) );

                CUdeviceptr hitgroup_record;
                size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
                OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &hitgroup_record ), hitgroup_record_size ) );
                RayGenSbtRecord hg_sbt;
                OTK_ERROR_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
                OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( hitgroup_record ), &hg_sbt,
                                               hitgroup_record_size, cudaMemcpyHostToDevice ) );

                sbt.raygenRecord                = raygen_record;
                sbt.missRecordBase              = miss_record;
                sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
                sbt.missRecordCount             = 1;
                sbt.hitgroupRecordBase          = hitgroup_record;
                sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
                sbt.hitgroupRecordCount         = 1;
            }

            // Create texture
            initTexture( m_textureWidth, m_textureHeight, addressMode );
            m_granularity = getGranularityForTileSize( m_tileWidth, m_tileHeight );

            // Copy inputs to device
            FootprintInputs* d_inputs;
            OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &d_inputs ), inputs.size() * sizeof( FootprintInputs ) ) );
            OTK_ERROR_CHECK( cudaMemcpy( d_inputs, inputs.data(), inputs.size() * sizeof( FootprintInputs ), cudaMemcpyHostToDevice ) );

            // Create output buffer.
            size_t numOutputs = inputs.size();
            uint4* d_outputs;
            OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &d_outputs ), 2 * numOutputs * sizeof( uint4 ) ) );

            // Create usage bits
            unsigned int  referenceBitsSizeInWords = demandLoading::MAX_TILE_LEVELS * MAX_PAGES_PER_MIP_LEVEL / 32;
            unsigned int  referenceBitsSizeInBytes = demandLoading::MAX_TILE_LEVELS * MAX_PAGES_PER_MIP_LEVEL / 8;
            unsigned int* d_referenceBits;
            OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &d_referenceBits ), referenceBitsSizeInBytes ) );
            OTK_ERROR_CHECK( cuMemsetD8( reinterpret_cast<CUdeviceptr>( d_referenceBits ), 0, referenceBitsSizeInBytes ) );

            unsigned int  residenceBitsSizeInBytes = referenceBitsSizeInBytes;
            unsigned int* d_residenceBits;
            OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &d_residenceBits ), residenceBitsSizeInBytes ) );
            OTK_ERROR_CHECK( cuMemsetD8( reinterpret_cast<CUdeviceptr>( d_residenceBits ), 0, residenceBitsSizeInBytes ) );

            //
            // Launch
            //
            Params params{};
            {
                CUstream stream;
                OTK_ERROR_CHECK( cudaStreamCreate( &stream ) );

                params.sampler.texture = m_texture;
                params.sampler.desc.numMipLevels =
                    static_cast<unsigned int>( log2( std::max( m_textureWidth, m_textureHeight ) ) );
                params.sampler.desc.logTileWidth = static_cast<unsigned int>( log2f( static_cast<float>( m_tileWidth ) ) );
                params.sampler.desc.logTileHeight = static_cast<unsigned int>( log2f( static_cast<float>( m_tileHeight ) ) );
                params.sampler.desc.isSparseTexture  = 1;
                params.sampler.desc.wrapMode0        = addressMode;
                params.sampler.desc.wrapMode1        = addressMode;
                params.sampler.desc.mipmapFilterMode = cudaFilterModeLinear;
                params.sampler.desc.maxAnisotropy    = 16;

                params.sampler.width  = m_textureWidth;
                params.sampler.height = m_textureHeight;

                unsigned int mipTailStartX = static_cast<unsigned int>( 1 + ceil( log2( m_textureWidth / m_tileWidth ) ) );
                unsigned int mipTailStartY = static_cast<unsigned int>( 1 + ceil( log2( m_textureHeight / m_tileHeight ) ) );
                m_mipTailFirstLevel        = std::max( mipTailStartX, mipTailStartY );
                params.sampler.mipTailFirstLevel = m_mipTailFirstLevel;

                params.sampler.startPage = 0;
                params.sampler.numPages  = demandLoading::MAX_TILE_LEVELS * MAX_PAGES_PER_MIP_LEVEL;

                params.referenceBits = d_referenceBits;
                params.residenceBits = d_residenceBits;
                params.inputs        = d_inputs;
                params.outputs       = d_outputs;


                for( unsigned int level = 0; level < demandLoading::MAX_TILE_LEVELS; ++level )
                {
                    if( level >= m_mipTailFirstLevel )
                        params.sampler.mipLevelSizes[level].mipLevelStart = 0;
                    else
                        params.sampler.mipLevelSizes[level].mipLevelStart =
                            MAX_PAGES_PER_MIP_LEVEL * ( m_mipTailFirstLevel - level );
                    params.sampler.mipLevelSizes[level].levelWidthInTiles =
                            static_cast<unsigned short>( demandLoading::getLevelDimInTiles( m_textureWidth, level, m_tileWidth ) );
                    params.sampler.mipLevelSizes[level].levelHeightInTiles =
                            static_cast<unsigned short>( demandLoading::getLevelDimInTiles( m_textureHeight, level, m_tileHeight ) );
                }

                CUdeviceptr d_params;
                OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &d_params ), sizeof( Params ) ) );
                OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_params ), &params, sizeof( params ), cudaMemcpyHostToDevice ) );

                OTK_ERROR_CHECK( optixLaunch( pipeline, stream, d_params, sizeof( Params ), &sbt,
                                          static_cast<unsigned int>( inputs.size() ), 1, 1 ) );
                OTK_CUDA_SYNC_CHECK();
                OTK_ERROR_CHECK( cuMemFree( d_params ) );
            }

            // Copy output to host (returned via result parameter)
            std::vector<uint4> outputs( 2 * numOutputs );
            OTK_ERROR_CHECK( cudaMemcpy( outputs.data(), d_outputs, 2 * numOutputs * sizeof( uint4 ), cudaMemcpyDeviceToHost ) );

            // Check results
            checkResults( inputs, outputs, expectedTileCoords );

            // Check reference bits
            std::vector<unsigned int> referenceBits( referenceBitsSizeInWords, 0 );
            OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( referenceBits.data() ), reinterpret_cast<void*>( d_referenceBits ),
                                           referenceBitsSizeInBytes, cudaMemcpyDeviceToHost ) );
            checkReferenceBits( expectedTileCoords, referenceBits, params.sampler );

            //
            // Cleanup
            //
            {
                destroyTexture();
                OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( d_inputs ) ) );
                OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( d_outputs ) ) );
                OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( d_referenceBits ) ) );
                OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( d_residenceBits ) ) );

                OTK_ERROR_CHECK( cuMemFree( sbt.raygenRecord ) );
                OTK_ERROR_CHECK( cuMemFree( sbt.missRecordBase ) );
                OTK_ERROR_CHECK( cuMemFree( sbt.hitgroupRecordBase ) );

                OTK_ERROR_CHECK( optixPipelineDestroy( pipeline ) );
                OTK_ERROR_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
                OTK_ERROR_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
                OTK_ERROR_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
                OTK_ERROR_CHECK( optixModuleDestroy( module ) );

                OTK_ERROR_CHECK( optixDeviceContextDestroy( context ) );
            }
        }
        catch( std::exception& e )
        {
            std::cerr << "Caught exception: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

  private:
    void fillBits( std::vector<unsigned int>& bits, TileCoords& tc, const demandLoading::TextureSampler& sampler )
    {
        unsigned int mipLevel = std::min( tc.mipLevel, m_mipTailFirstLevel );
        for( unsigned int i = 0; i < tc.numCoords; ++i )
        {
            unsigned int levelWidthInTiles = demandLoading::getLevelDimInTiles( m_textureWidth, tc.mipLevel, m_tileWidth );
            unsigned int mipLevelStart     = sampler.mipLevelSizes[mipLevel].mipLevelStart;
            unsigned int idx =
                mipLevelStart + demandLoading::getPageOffsetFromTileCoords( tc.coords[i].x, tc.coords[i].y, levelWidthInTiles );
            ASSERT_TRUE( idx < static_cast<unsigned int>( bits.size() * 32 ) );
            bits[idx / 32] = bits[idx / 32] | ( 1 << ( idx % 32 ) );
        }
    }

    void checkReferenceBits( const std::vector<TileCoords>&       expectedTileCoords,
                             const std::vector<unsigned int>&     referenceBits,
                             const demandLoading::TextureSampler& sampler )
    {
        std::vector<unsigned int> expectedUsageBits( referenceBits.size(), 0 );
        for( TileCoords tc : expectedTileCoords )
        {
            fillBits( expectedUsageBits, tc, sampler );
        }

        for( unsigned int i = 0; i < static_cast<unsigned int>( referenceBits.size() ); ++i )
        {
            if( referenceBits[i] != expectedUsageBits[i] )
            {
                printf( "Bit mismatch at index %d (bit index %d): expected:%x, actual:%x\n", i, i * 32,
                        expectedUsageBits[i], referenceBits[i] );

                // Check that the returned referenceBits are a superset of the expectedUsageBits.
                // This handles the approximations in the software version of the footprint instruction.
                EXPECT_EQ( expectedUsageBits[i], ( referenceBits[i] & expectedUsageBits[i] ) );
            }
        }
    }

    void checkResults( const std::vector<FootprintInputs>& inputs,
                       const std::vector<uint4>&           outputs,
                       const std::vector<TileCoords>&      expectedTileCoords ) const
    {
        ASSERT_EQ( inputs.size() * 2, outputs.size() );

        // Convert footprint results to tile coordinates.
        std::vector<TileCoords> tileCoords;
        tileCoords.reserve( outputs.size() );

        for( unsigned int i = 0; i < static_cast<unsigned int>( expectedTileCoords.size() ); ++i )
        {
            // Unpack footprint to tile coordinates
            const demandLoading::Texture2DFootprint* fp =
                reinterpret_cast<const demandLoading::Texture2DFootprint*>( &outputs[i] );
            tileCoords.push_back( getTileCoordsFromFootprint( *fp ) );
            unsigned int levelWidthInTiles = demandLoading::getLevelDimInTiles( m_textureWidth, fp->level, m_tileWidth );
            unsigned int levelHeightInTiles = demandLoading::getLevelDimInTiles( m_textureHeight, fp->level, m_tileHeight );
            wrapTileCoords( tileCoords.back(), levelWidthInTiles, levelHeightInTiles );
        }

        // Compare to expected results.
        bool ok = tileCoords.size() == expectedTileCoords.size();
        for( size_t i = 0; i < tileCoords.size() && i < expectedTileCoords.size(); ++i )
        {
            if( !isSuperset( tileCoords[i], expectedTileCoords[i] ) )
            {
                std::cerr << "Expected: " << expectedTileCoords[i] << std::endl;
                std::cerr << "  Actual: " << tileCoords[i] << std::endl;
            }

            // Check that the returned tile coords are a superset of the expected tile coords.
            // This handles the approximations in the software version of the footprint instruction.
            ok = ok && isSuperset( tileCoords[i], expectedTileCoords[i] );
            EXPECT_TRUE( isSuperset( tileCoords[i], expectedTileCoords[i] ) );
        }
        // If there's a mismatch, dump all outputs.
        if( !ok )
        {
            std::cerr << "Actual tile coordinates: {\n";
            for( size_t i = 0; i < tileCoords.size(); ++i )
            {
                if( i > 0 )
                    std::cerr << ", ";
                std::cerr << tileCoords[i];
            }
            std::cerr << "\n}\n";
        }
    }
};


class TextureFootprintTest : public testing::Test
{
  public:
    void SetUp() override { m_fixture.reset( new TextureFootprintFixture ); }

    std::unique_ptr<TextureFootprintFixture> m_fixture;
};


}  // namespace

//------------------------------------------------------------------------------
// Test Instances

TEST_F( TextureFootprintTest, TestCornersAndEdges )
{
    // Corners and edges of the textures space
    std::vector<FootprintInputs> inputs{FootprintInputs( 0.f, 0.f ),    FootprintInputs( 1.f, 0.f ),
                                        FootprintInputs( 0.f, 1.f ),    FootprintInputs( 1.f, 1.f ),
                                        FootprintInputs( 0.25f, 0.0f ), FootprintInputs( 0.25f, 1.0f ),
                                        FootprintInputs( 0.0f, 0.25f ), FootprintInputs( 1.0f, 0.25f )};

    // Run test
    std::vector<TileCoords> expectedTileCoords{{0, 1, {{0, 0}}},         {0, 1, {{15, 0}}},
                                               {0, 1, {{0, 15}}},        {0, 1, {{15, 15}}},
                                               {0, 2, {{3, 0}, {4, 0}}}, {0, 2, {{3, 15}, {4, 15}}},
                                               {0, 2, {{0, 3}, {0, 4}}}, {0, 2, {{15, 3}, {15, 4}}}};
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprint", inputs, expectedTileCoords, cudaAddressModeClamp ) );
}

TEST_F( TextureFootprintTest, TestSingleBlock )
{
    // Block centers
    std::vector<FootprintInputs> inputs{{0.25f, 0.25f}, {0.75f, 0.25f}, {0.25f, 0.75f}};

    // Run test
    std::vector<TileCoords> expectedTileCoords{{0, 4, {{3, 3}, {4, 3}, {3, 4}, {4, 4}}},
                                               {0, 4, {{11, 3}, {12, 3}, {11, 4}, {12, 4}}},
                                               {0, 4, {{3, 11}, {4, 11}, {3, 12}, {4, 12}}}};
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprint", inputs, expectedTileCoords, cudaAddressModeClamp ) );
}

TEST_F( TextureFootprintTest, TestSpanTiles )
{
    // Spanning four tiles, or two tiles vertically and horizontally
    std::vector<FootprintInputs> inputs{{0.5f, 0.5f}, {0.49f, 0.5f}, {0.5f, 0.49f}};

    // Run test
    std::vector<TileCoords> expectedTileCoords{
        {0, 4, {{7, 7}, {8, 7}, {8, 8}, {7, 8}}}, {0, 2, {{7, 7}, {7, 8}}}, {0, 2, {{7, 7}, {8, 7}}}};
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprint", inputs, expectedTileCoords, cudaAddressModeClamp ) );
}

TEST_F( TextureFootprintTest, TestSlightOffsetsFromTileCorners )
{
    // Tile corners with slight offsets +/-
    std::vector<FootprintInputs> inputs{{0.4999f, 0.4999f}, {0.4999f, 0.5001f}, {0.5001f, 0.4999f}, {0.5001f, 0.5001f},
                                        {0.4374f, 0.4374f}, {0.4374f, 0.4376f}, {0.4376f, 0.4374f}, {0.4376f, 0.4376f}};

    // Run test
    std::vector<TileCoords> expectedTileCoords{
        {0, 4, {{7, 7}, {8, 7}, {8, 8}, {7, 8}}}, {0, 4, {{7, 7}, {8, 7}, {8, 8}, {7, 8}}},
        {0, 4, {{7, 7}, {8, 7}, {8, 8}, {7, 8}}}, {0, 4, {{7, 7}, {8, 7}, {8, 8}, {7, 8}}},
        {0, 4, {{6, 6}, {7, 6}, {6, 7}, {7, 7}}}, {0, 4, {{6, 6}, {7, 6}, {6, 7}, {7, 7}}},
        {0, 4, {{6, 6}, {7, 6}, {6, 7}, {7, 7}}}, {0, 4, {{6, 6}, {7, 6}, {6, 7}, {7, 7}}}};
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprint", inputs, expectedTileCoords, cudaAddressModeClamp ) );
}

TEST_F( TextureFootprintTest, TestLodPoints )
{
    // Tile corners at one level that are in the middle of a tile in the next level
    std::vector<FootprintInputs> inputs;
    inputs.push_back( FootprintInputs( 0.0625f, 0.0625f, 0.0f ) );
    inputs.push_back( FootprintInputs( 0.0625f, 0.0625f, 1.0f ) );
    inputs.push_back( FootprintInputs( 0.5f, 0.5f, 3.0f ) );
    inputs.push_back( FootprintInputs( 0.5f, 0.5f, 4.0f ) );

    // Run test.
    std::vector<TileCoords> expectedTileCoords{
        {0, 4, {{0, 0}, {1, 0}, {0, 1}, {1, 1}}}, {1, 1, {{0, 0}}}, {3, 4, {{0, 0}, {1, 0}, {0, 1}, {1, 1}}}, {4, 1, {{0, 0}}}};
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprintLod", inputs, expectedTileCoords, cudaAddressModeClamp ) );
}

TEST_F( TextureFootprintTest, TestGradLevels )
{
    // Test texture LODs resulting from different gradient lengths
    std::vector<FootprintInputs> inputs;
    const float                  dummyLod = 0.0f;
    inputs.push_back( FootprintInputs( 0.03125f, 0.03125f, dummyLod, 1.0f / 1000.0f, 0.0f, 0.0f, 1.0f / 1000.0f ) );
    inputs.push_back( FootprintInputs( 0.03125f, 0.03125f, dummyLod, 1.0f / 500.0f, 0.0f, 0.0f, 1.0f / 500.0f ) );
    inputs.push_back( FootprintInputs( 0.03125f, 0.03125f, dummyLod, 1.0f / 250.0f, 0.0f, 0.0f, 1.0f / 250.0f ) );
    inputs.push_back( FootprintInputs( 0.03125f, 0.03125f, dummyLod, 1.0f / 125.0f, 0.0f, 0.0f, 1.0f / 125.0f ) );
    inputs.push_back( FootprintInputs( 0.03125f, 0.03125f, dummyLod, 1.0f / 62.0f, 0.0f, 0.0f, 1.0f / 62.0f ) );
    inputs.push_back( FootprintInputs( 0.03125f, 0.03125f, dummyLod, 1.0f / 31.0f, 0.0f, 0.0f, 1.0f / 31.0f ) );
    inputs.push_back( FootprintInputs( 0.03125f, 0.03125f, dummyLod, 1.1f, 0.0f, 0.0f, 1.1f ) );

    // Run test.
    std::vector<TileCoords> expectedTileCoords{{0, 1, {{0, 0}}}, {1, 1, {{0, 0}}}, {2, 1, {{0, 0}}}, {3, 1, {{0, 0}}},
                                               {4, 1, {{0, 0}}}, {5, 1, {{0, 0}}}, {10, 1, {{0, 0}}}};
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprintGrad", inputs, expectedTileCoords, cudaAddressModeClamp ) );
}

TEST_F( TextureFootprintTest, TestGradDirections )
{
    // Gradients in different directions spanning tiles
    std::vector<FootprintInputs> inputs;
    const float                  dummyLod = 0.0f;
    inputs.push_back( FootprintInputs(0.497f, 0.497f, dummyLod, 1.0f / 1000.0f, 0.0f, 0.0f, 1.0f / 1000.0f) );
    inputs.push_back( FootprintInputs( 0.497f, 0.497f, dummyLod, 8.0f / 1000.0f, 0.0f, 0.0f, 1.0f / 1000.0f ) );
    inputs.push_back( FootprintInputs( 0.497f, 0.497f, dummyLod, 1.0f / 1000.0f, 0.0f, 0.0f, 8.0f / 1000.0f ) );
    inputs.push_back( FootprintInputs( 0.497f, 0.497f, dummyLod, 8.0f / 1000.0f, 8.0f / 1000.0f, 0.0f, 0.0f ) );
    inputs.push_back( FootprintInputs( 0.497f, 0.497f, dummyLod, 8.0f / 1000.0f, -8.0f / 1000.0f, 0.0f, 0.0f ) );
    inputs.push_back( FootprintInputs( 0.497f, 0.497f, dummyLod, 32.0f / 1000.0f, 0.0f, 0.0f, 0.0f ) );

    // Run test.
    std::vector<TileCoords> expectedTileCoords{{0, 1, {{7, 7}}},
                                               {0, 2, {{7, 7}, {8, 7}}},
                                               {0, 2, {{7, 7}, {7, 8}}},
                                               {0, 4, {{7, 7}, {8, 7}, {8, 8}, {7, 8}}},
                                               {0, 3, {{7, 7}, {8, 7}, {7, 8}}},
                                               {1, 2, {{3, 3}, {4, 3}}}};
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprintGrad", inputs, expectedTileCoords, cudaAddressModeClamp ) );
}

TEST_F( TextureFootprintTest, TestOnGrid )
{
    // Create inputs.
    const unsigned int           launchWidth  = 17;
    const unsigned int           launchHeight = 17;
    std::vector<FootprintInputs> inputs;
    inputs.reserve( launchWidth * launchHeight );
    for( unsigned int j = 0; j < launchHeight; ++j )
    {
        for( unsigned int i = 0; i < launchWidth; ++i )
        {
            float x = static_cast<float>( i ) / launchWidth;
            float y = static_cast<float>( j ) / launchHeight;
            inputs.push_back( FootprintInputs( x, y ) );
        }
    }

    // Run test.
    std::vector<TileCoords> expectedTileCoords{
        {0, 1, {{0, 0}}},   {0, 1, {{0, 0}}},   {0, 1, {{1, 0}}},   {0, 1, {{2, 0}}},   {0, 1, {{3, 0}}},
        {0, 1, {{4, 0}}},   {0, 1, {{5, 0}}},   {0, 1, {{6, 0}}},   {0, 1, {{7, 0}}},   {0, 1, {{8, 0}}},
        {0, 1, {{9, 0}}},   {0, 1, {{10, 0}}},  {0, 1, {{11, 0}}},  {0, 1, {{12, 0}}},  {0, 1, {{13, 0}}},  // 10
        {0, 1, {{14, 0}}},  {0, 1, {{15, 0}}},  {0, 1, {{0, 0}}},   {0, 1, {{0, 0}}},   {0, 1, {{1, 0}}},
        {0, 1, {{2, 0}}},   {0, 1, {{3, 0}}},   {0, 1, {{4, 0}}},   {0, 1, {{5, 0}}},   {0, 1, {{6, 0}}},  // 20
        {0, 1, {{7, 0}}},   {0, 1, {{8, 0}}},   {0, 1, {{9, 0}}},   {0, 1, {{10, 0}}},  {0, 1, {{11, 0}}},
        {0, 1, {{12, 0}}},  {0, 1, {{13, 0}}},  {0, 1, {{14, 0}}},  {0, 1, {{15, 0}}},  {0, 1, {{0, 1}}},  // 30
        {0, 1, {{0, 1}}},   {0, 1, {{1, 1}}},   {0, 1, {{2, 1}}},   {0, 1, {{3, 1}}},   {0, 1, {{4, 1}}},
        {0, 1, {{5, 1}}},   {0, 1, {{6, 1}}},   {0, 1, {{7, 1}}},   {0, 1, {{8, 1}}},   {0, 1, {{9, 1}}},  // 40
        {0, 1, {{10, 1}}},  {0, 1, {{11, 1}}},  {0, 1, {{12, 1}}},  {0, 1, {{13, 1}}},  {0, 1, {{14, 1}}},
        {0, 1, {{15, 1}}},  {0, 1, {{0, 2}}},   {0, 1, {{0, 2}}},   {0, 1, {{1, 2}}},   {0, 1, {{2, 2}}},  // 50
        {0, 1, {{3, 2}}},   {0, 1, {{4, 2}}},   {0, 1, {{5, 2}}},   {0, 1, {{6, 2}}},   {0, 1, {{7, 2}}},
        {0, 1, {{8, 2}}},   {0, 1, {{9, 2}}},   {0, 1, {{10, 2}}},  {0, 1, {{11, 2}}},  {0, 1, {{12, 2}}},  // 60
        {0, 1, {{13, 2}}},  {0, 1, {{14, 2}}},  {0, 1, {{15, 2}}},  {0, 1, {{0, 3}}},   {0, 1, {{0, 3}}},
        {0, 1, {{1, 3}}},   {0, 1, {{2, 3}}},   {0, 1, {{3, 3}}},   {0, 1, {{4, 3}}},   {0, 1, {{5, 3}}},  // 70
        {0, 1, {{6, 3}}},   {0, 1, {{7, 3}}},   {0, 1, {{8, 3}}},   {0, 1, {{9, 3}}},   {0, 1, {{10, 3}}},
        {0, 1, {{11, 3}}},  {0, 1, {{12, 3}}},  {0, 1, {{13, 3}}},  {0, 1, {{14, 3}}},  {0, 1, {{15, 3}}},  // 80
        {0, 1, {{0, 4}}},   {0, 1, {{0, 4}}},   {0, 1, {{1, 4}}},   {0, 1, {{2, 4}}},   {0, 1, {{3, 4}}},
        {0, 1, {{4, 4}}},   {0, 1, {{5, 4}}},   {0, 1, {{6, 4}}},   {0, 1, {{7, 4}}},   {0, 1, {{8, 4}}},  // 90
        {0, 1, {{9, 4}}},   {0, 1, {{10, 4}}},  {0, 1, {{11, 4}}},  {0, 1, {{12, 4}}},  {0, 1, {{13, 4}}},
        {0, 1, {{14, 4}}},  {0, 1, {{15, 4}}},  {0, 1, {{0, 5}}},   {0, 1, {{0, 5}}},   {0, 1, {{1, 5}}},  // 100
        {0, 1, {{2, 5}}},   {0, 1, {{3, 5}}},   {0, 1, {{4, 5}}},   {0, 1, {{5, 5}}},   {0, 1, {{6, 5}}},
        {0, 1, {{7, 5}}},   {0, 1, {{8, 5}}},   {0, 1, {{9, 5}}},   {0, 1, {{10, 5}}},  {0, 1, {{11, 5}}},  // 110
        {0, 1, {{12, 5}}},  {0, 1, {{13, 5}}},  {0, 1, {{14, 5}}},  {0, 1, {{15, 5}}},  {0, 1, {{0, 6}}},
        {0, 1, {{0, 6}}},   {0, 1, {{1, 6}}},   {0, 1, {{2, 6}}},   {0, 1, {{3, 6}}},   {0, 1, {{4, 6}}},  // 120
        {0, 1, {{5, 6}}},   {0, 1, {{6, 6}}},   {0, 1, {{7, 6}}},   {0, 1, {{8, 6}}},   {0, 1, {{9, 6}}},
        {0, 1, {{10, 6}}},  {0, 1, {{11, 6}}},  {0, 1, {{12, 6}}},  {0, 1, {{13, 6}}},  {0, 1, {{14, 6}}},  // 130
        {0, 1, {{15, 6}}},  {0, 1, {{0, 7}}},   {0, 1, {{0, 7}}},   {0, 1, {{1, 7}}},   {0, 1, {{2, 7}}},
        {0, 1, {{3, 7}}},   {0, 1, {{4, 7}}},   {0, 1, {{5, 7}}},   {0, 1, {{6, 7}}},   {0, 1, {{7, 7}}},  // 140
        {0, 1, {{8, 7}}},   {0, 1, {{9, 7}}},   {0, 1, {{10, 7}}},  {0, 1, {{11, 7}}},  {0, 1, {{12, 7}}},
        {0, 1, {{13, 7}}},  {0, 1, {{14, 7}}},  {0, 1, {{15, 7}}},  {0, 1, {{0, 8}}},   {0, 1, {{0, 8}}},  // 150
        {0, 1, {{1, 8}}},   {0, 1, {{2, 8}}},   {0, 1, {{3, 8}}},   {0, 1, {{4, 8}}},   {0, 1, {{5, 8}}},
        {0, 1, {{6, 8}}},   {0, 1, {{7, 8}}},   {0, 1, {{8, 8}}},   {0, 1, {{9, 8}}},   {0, 1, {{10, 8}}},  // 160
        {0, 1, {{11, 8}}},  {0, 1, {{12, 8}}},  {0, 1, {{13, 8}}},  {0, 1, {{14, 8}}},  {0, 1, {{15, 8}}},
        {0, 1, {{0, 9}}},   {0, 1, {{0, 9}}},   {0, 1, {{1, 9}}},   {0, 1, {{2, 9}}},   {0, 1, {{3, 9}}},  // 170
        {0, 1, {{4, 9}}},   {0, 1, {{5, 9}}},   {0, 1, {{6, 9}}},   {0, 1, {{7, 9}}},   {0, 1, {{8, 9}}},
        {0, 1, {{9, 9}}},   {0, 1, {{10, 9}}},  {0, 1, {{11, 9}}},  {0, 1, {{12, 9}}},  {0, 1, {{13, 9}}},  // 180
        {0, 1, {{14, 9}}},  {0, 1, {{15, 9}}},  {0, 1, {{0, 10}}},  {0, 1, {{0, 10}}},  {0, 1, {{1, 10}}},
        {0, 1, {{2, 10}}},  {0, 1, {{3, 10}}},  {0, 1, {{4, 10}}},  {0, 1, {{5, 10}}},  {0, 1, {{6, 10}}},  // 190
        {0, 1, {{7, 10}}},  {0, 1, {{8, 10}}},  {0, 1, {{9, 10}}},  {0, 1, {{10, 10}}}, {0, 1, {{11, 10}}},
        {0, 1, {{12, 10}}}, {0, 1, {{13, 10}}}, {0, 1, {{14, 10}}}, {0, 1, {{15, 10}}}, {0, 1, {{0, 11}}},  // 200
        {0, 1, {{0, 11}}},  {0, 1, {{1, 11}}},  {0, 1, {{2, 11}}},  {0, 1, {{3, 11}}},  {0, 1, {{4, 11}}},
        {0, 1, {{5, 11}}},  {0, 1, {{6, 11}}},  {0, 1, {{7, 11}}},  {0, 1, {{8, 11}}},  {0, 1, {{9, 11}}},  // 210
        {0, 1, {{10, 11}}}, {0, 1, {{11, 11}}}, {0, 1, {{12, 11}}}, {0, 1, {{13, 11}}}, {0, 1, {{14, 11}}},
        {0, 1, {{15, 11}}}, {0, 1, {{0, 12}}},  {0, 1, {{0, 12}}},  {0, 1, {{1, 12}}},  {0, 1, {{2, 12}}},  // 220
        {0, 1, {{3, 12}}},  {0, 1, {{4, 12}}},  {0, 1, {{5, 12}}},  {0, 1, {{6, 12}}},  {0, 1, {{7, 12}}},
        {0, 1, {{8, 12}}},  {0, 1, {{9, 12}}},  {0, 1, {{10, 12}}}, {0, 1, {{11, 12}}}, {0, 1, {{12, 12}}},  // 230
        {0, 1, {{13, 12}}}, {0, 1, {{14, 12}}}, {0, 1, {{15, 12}}}, {0, 1, {{0, 13}}},  {0, 1, {{0, 13}}},
        {0, 1, {{1, 13}}},  {0, 1, {{2, 13}}},  {0, 1, {{3, 13}}},  {0, 1, {{4, 13}}},  {0, 1, {{5, 13}}},  // 240
        {0, 1, {{6, 13}}},  {0, 1, {{7, 13}}},  {0, 1, {{8, 13}}},  {0, 1, {{9, 13}}},  {0, 1, {{10, 13}}},
        {0, 1, {{11, 13}}}, {0, 1, {{12, 13}}}, {0, 1, {{13, 13}}}, {0, 1, {{14, 13}}}, {0, 1, {{15, 13}}},  // 250
        {0, 1, {{0, 14}}},  {0, 1, {{0, 14}}},  {0, 1, {{1, 14}}},  {0, 1, {{2, 14}}},  {0, 1, {{3, 14}}},
        {0, 1, {{4, 14}}},  {0, 1, {{5, 14}}},  {0, 1, {{6, 14}}},  {0, 1, {{7, 14}}},  {0, 1, {{8, 14}}},  // 260
        {0, 1, {{9, 14}}},  {0, 1, {{10, 14}}}, {0, 1, {{11, 14}}}, {0, 1, {{12, 14}}}, {0, 1, {{13, 14}}},
        {0, 1, {{14, 14}}}, {0, 1, {{15, 14}}}, {0, 1, {{0, 15}}},  {0, 1, {{0, 15}}},  {0, 1, {{1, 15}}},  // 270
        {0, 1, {{2, 15}}},  {0, 1, {{3, 15}}},  {0, 1, {{4, 15}}},  {0, 1, {{5, 15}}},  {0, 1, {{6, 15}}},
        {0, 1, {{7, 15}}},  {0, 1, {{8, 15}}},  {0, 1, {{9, 15}}},  {0, 1, {{10, 15}}}, {0, 1, {{11, 15}}},  // 280
        {0, 1, {{12, 15}}}, {0, 1, {{13, 15}}}, {0, 1, {{14, 15}}}, {0, 1, {{15, 15}}}};

    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprint", inputs, expectedTileCoords, cudaAddressModeClamp ) );
}

TEST_F( TextureFootprintTest, TestLodOnGrid )
{
    // Create inputs.
    const unsigned int           maxLod       = 10;
    const unsigned int           launchWidth  = 17;
    const unsigned int           launchHeight = 17;
    std::vector<FootprintInputs> inputs;
    inputs.reserve( launchWidth * launchHeight );
    for( unsigned int j = 0; j < launchHeight; ++j )
    {
        for( unsigned int i = 0; i < launchWidth; ++i )
        {
            float x   = static_cast<float>( i ) / launchWidth;
            float y   = static_cast<float>( j ) / launchHeight;
            float lod = x * x * x * x * maxLod;
            lod       = floorf( lod + 0.25f );
            inputs.push_back( FootprintInputs( x, y, lod ) );
        }
    }

    // Run test.
    std::vector<TileCoords> expectedTileCoords{
        {0, 1, {{0, 0}}},  {0, 1, {{0, 0}}},  {0, 1, {{1, 0}}},  {0, 1, {{2, 0}}},  {0, 1, {{3, 0}}},
        {0, 1, {{4, 0}}},  {0, 1, {{5, 0}}},  {0, 1, {{6, 0}}},  {0, 1, {{7, 0}}},  {1, 1, {{4, 0}}},
        {1, 1, {{4, 0}}},  {2, 1, {{2, 0}}},  {2, 1, {{2, 0}}},  {3, 1, {{1, 0}}},  {4, 1, {{0, 0}}},
        {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}},  {0, 1, {{0, 0}}},  {0, 1, {{0, 0}}},  {0, 1, {{1, 0}}},
        {0, 1, {{2, 0}}},  {0, 1, {{3, 0}}},  {0, 1, {{4, 0}}},  {0, 1, {{5, 0}}},  {0, 1, {{6, 0}}},
        {0, 1, {{7, 0}}},  {1, 1, {{4, 0}}},  {1, 1, {{4, 0}}},  {2, 1, {{2, 0}}},  {2, 1, {{2, 0}}},
        {3, 1, {{1, 0}}},  {4, 1, {{0, 0}}},  {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}},  {0, 1, {{0, 1}}},
        {0, 1, {{0, 1}}},  {0, 1, {{1, 1}}},  {0, 1, {{2, 1}}},  {0, 1, {{3, 1}}},  {0, 1, {{4, 1}}},
        {0, 1, {{5, 1}}},  {0, 1, {{6, 1}}},  {0, 1, {{7, 1}}},  {1, 1, {{4, 0}}},  {1, 1, {{4, 0}}},
        {2, 1, {{2, 0}}},  {2, 1, {{2, 0}}},  {3, 1, {{1, 0}}},  {4, 1, {{0, 0}}},  {6, 1, {{0, 0}}},
        {8, 1, {{0, 0}}},  {0, 1, {{0, 2}}},  {0, 1, {{0, 2}}},  {0, 1, {{1, 2}}},  {0, 1, {{2, 2}}},
        {0, 1, {{3, 2}}},  {0, 1, {{4, 2}}},  {0, 1, {{5, 2}}},  {0, 1, {{6, 2}}},  {0, 1, {{7, 2}}},
        {1, 1, {{4, 1}}},  {1, 1, {{4, 1}}},  {2, 1, {{2, 0}}},  {2, 1, {{2, 0}}},  {3, 1, {{1, 0}}},
        {4, 1, {{0, 0}}},  {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}},  {0, 1, {{0, 3}}},  {0, 1, {{0, 3}}},
        {0, 1, {{1, 3}}},  {0, 1, {{2, 3}}},  {0, 1, {{3, 3}}},  {0, 1, {{4, 3}}},  {0, 1, {{5, 3}}},
        {0, 1, {{6, 3}}},  {0, 1, {{7, 3}}},  {1, 1, {{4, 1}}},  {1, 1, {{4, 1}}},  {2, 1, {{2, 0}}},
        {2, 1, {{2, 0}}},  {3, 1, {{1, 0}}},  {4, 1, {{0, 0}}},  {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}},
        {0, 1, {{0, 4}}},  {0, 1, {{0, 4}}},  {0, 1, {{1, 4}}},  {0, 1, {{2, 4}}},  {0, 1, {{3, 4}}},
        {0, 1, {{4, 4}}},  {0, 1, {{5, 4}}},  {0, 1, {{6, 4}}},  {0, 1, {{7, 4}}},  {1, 1, {{4, 2}}},
        {1, 1, {{4, 2}}},  {2, 1, {{2, 1}}},  {2, 1, {{2, 1}}},  {3, 1, {{1, 0}}},  {4, 1, {{0, 0}}},
        {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}},  {0, 1, {{0, 5}}},  {0, 1, {{0, 5}}},  {0, 1, {{1, 5}}},
        {0, 1, {{2, 5}}},  {0, 1, {{3, 5}}},  {0, 1, {{4, 5}}},  {0, 1, {{5, 5}}},  {0, 1, {{6, 5}}},
        {0, 1, {{7, 5}}},  {1, 1, {{4, 2}}},  {1, 1, {{4, 2}}},  {2, 1, {{2, 1}}},  {2, 1, {{2, 1}}},
        {3, 1, {{1, 0}}},  {4, 1, {{0, 0}}},  {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}},  {0, 1, {{0, 6}}},
        {0, 1, {{0, 6}}},  {0, 1, {{1, 6}}},  {0, 1, {{2, 6}}},  {0, 1, {{3, 6}}},  {0, 1, {{4, 6}}},
        {0, 1, {{5, 6}}},  {0, 1, {{6, 6}}},  {0, 1, {{7, 6}}},  {1, 1, {{4, 3}}},  {1, 1, {{4, 3}}},
        {2, 1, {{2, 1}}},  {2, 1, {{2, 1}}},  {3, 1, {{1, 0}}},  {4, 1, {{0, 0}}},  {6, 1, {{0, 0}}},
        {8, 1, {{0, 0}}},  {0, 1, {{0, 7}}},  {0, 1, {{0, 7}}},  {0, 1, {{1, 7}}},  {0, 1, {{2, 7}}},
        {0, 1, {{3, 7}}},  {0, 1, {{4, 7}}},  {0, 1, {{5, 7}}},  {0, 1, {{6, 7}}},  {0, 1, {{7, 7}}},
        {1, 1, {{4, 3}}},  {1, 1, {{4, 3}}},  {2, 1, {{2, 1}}},  {2, 1, {{2, 1}}},  {3, 1, {{1, 0}}},
        {4, 1, {{0, 0}}},  {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}},  {0, 1, {{0, 8}}},  {0, 1, {{0, 8}}},
        {0, 1, {{1, 8}}},  {0, 1, {{2, 8}}},  {0, 1, {{3, 8}}},  {0, 1, {{4, 8}}},  {0, 1, {{5, 8}}},
        {0, 1, {{6, 8}}},  {0, 1, {{7, 8}}},  {1, 1, {{4, 4}}},  {1, 1, {{4, 4}}},  {2, 1, {{2, 2}}},
        {2, 1, {{2, 2}}},  {3, 1, {{1, 1}}},  {4, 1, {{0, 0}}},  {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}},
        {0, 1, {{0, 9}}},  {0, 1, {{0, 9}}},  {0, 1, {{1, 9}}},  {0, 1, {{2, 9}}},  {0, 1, {{3, 9}}},
        {0, 1, {{4, 9}}},  {0, 1, {{5, 9}}},  {0, 1, {{6, 9}}},  {0, 1, {{7, 9}}},  {1, 1, {{4, 4}}},
        {1, 1, {{4, 4}}},  {2, 1, {{2, 2}}},  {2, 1, {{2, 2}}},  {3, 1, {{1, 1}}},  {4, 1, {{0, 0}}},
        {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}},  {0, 1, {{0, 10}}}, {0, 1, {{0, 10}}}, {0, 1, {{1, 10}}},
        {0, 1, {{2, 10}}}, {0, 1, {{3, 10}}}, {0, 1, {{4, 10}}}, {0, 1, {{5, 10}}}, {0, 1, {{6, 10}}},
        {0, 1, {{7, 10}}}, {1, 1, {{4, 5}}},  {1, 1, {{4, 5}}},  {2, 1, {{2, 2}}},  {2, 1, {{2, 2}}},
        {3, 1, {{1, 1}}},  {4, 1, {{0, 0}}},  {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}},  {0, 1, {{0, 11}}},
        {0, 1, {{0, 11}}}, {0, 1, {{1, 11}}}, {0, 1, {{2, 11}}}, {0, 1, {{3, 11}}}, {0, 1, {{4, 11}}},
        {0, 1, {{5, 11}}}, {0, 1, {{6, 11}}}, {0, 1, {{7, 11}}}, {1, 1, {{4, 5}}},  {1, 1, {{4, 5}}},
        {2, 1, {{2, 2}}},  {2, 1, {{2, 2}}},  {3, 1, {{1, 1}}},  {4, 1, {{0, 0}}},  {6, 1, {{0, 0}}},
        {8, 1, {{0, 0}}},  {0, 1, {{0, 12}}}, {0, 1, {{0, 12}}}, {0, 1, {{1, 12}}}, {0, 1, {{2, 12}}},
        {0, 1, {{3, 12}}}, {0, 1, {{4, 12}}}, {0, 1, {{5, 12}}}, {0, 1, {{6, 12}}}, {0, 1, {{7, 12}}},
        {1, 1, {{4, 6}}},  {1, 1, {{4, 6}}},  {2, 1, {{2, 3}}},  {2, 1, {{2, 3}}},  {3, 1, {{1, 1}}},
        {4, 1, {{0, 0}}},  {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}},  {0, 1, {{0, 13}}}, {0, 1, {{0, 13}}},
        {0, 1, {{1, 13}}}, {0, 1, {{2, 13}}}, {0, 1, {{3, 13}}}, {0, 1, {{4, 13}}}, {0, 1, {{5, 13}}},
        {0, 1, {{6, 13}}}, {0, 1, {{7, 13}}}, {1, 1, {{4, 6}}},  {1, 1, {{4, 6}}},  {2, 1, {{2, 3}}},
        {2, 1, {{2, 3}}},  {3, 1, {{1, 1}}},  {4, 1, {{0, 0}}},  {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}},
        {0, 1, {{0, 14}}}, {0, 1, {{0, 14}}}, {0, 1, {{1, 14}}}, {0, 1, {{2, 14}}}, {0, 1, {{3, 14}}},
        {0, 1, {{4, 14}}}, {0, 1, {{5, 14}}}, {0, 1, {{6, 14}}}, {0, 1, {{7, 14}}}, {1, 1, {{4, 7}}},
        {1, 1, {{4, 7}}},  {2, 1, {{2, 3}}},  {2, 1, {{2, 3}}},  {3, 1, {{1, 1}}},  {4, 1, {{0, 0}}},
        {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}},  {0, 1, {{0, 15}}}, {0, 1, {{0, 15}}}, {0, 1, {{1, 15}}},
        {0, 1, {{2, 15}}}, {0, 1, {{3, 15}}}, {0, 1, {{4, 15}}}, {0, 1, {{5, 15}}}, {0, 1, {{6, 15}}},
        {0, 1, {{7, 15}}}, {1, 1, {{4, 7}}},  {1, 1, {{4, 7}}},  {2, 1, {{2, 3}}},  {2, 1, {{2, 3}}},
        {3, 1, {{1, 1}}},  {4, 1, {{0, 0}}},  {6, 1, {{0, 0}}},  {8, 1, {{0, 0}}}};
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprintLod", inputs, expectedTileCoords, cudaAddressModeClamp ) );
}

TEST_F( TextureFootprintTest, TestGradOnGrid )
{
    // Create inputs.
    const unsigned int           launchWidth  = 17;
    const unsigned int           launchHeight = 17;
    std::vector<FootprintInputs> inputs;
    inputs.reserve( launchWidth * launchHeight );
    for( unsigned int j = 0; j < launchHeight; ++j )
    {
        for( unsigned int i = 0; i < launchWidth; ++i )
        {
            float x      = static_cast<float>( i + 0.5f ) / launchWidth;
            float y      = static_cast<float>( j + 0.5f ) / launchHeight;
            float dPdx_x = x * x * x * x;
            float dPdx_y = 0;
            float dPdy_x = 0;
            float dPdy_y = y * y * y * y;

            // Make sure gradients are not too narrow
            dPdx_x = std::max( dPdx_x, dPdy_y / 8.0f );
            dPdy_y = std::max( dPdy_y, dPdx_x / 8.0f );

            // Scale the gradients to keep mip level in between levels to maintain consistency between
            // HW and SW footprint code. The workaround for this effect is tested in rendering tests.
            float filterWidth  = std::min( dPdx_x, dPdy_y );
            float mipLevel     = log2( filterWidth );
            float fracMipLevel = mipLevel - floorf( mipLevel );
            float mipScale     = exp2( 0.5f - fracMipLevel );
            dPdx_x *= mipScale;
            dPdy_y *= mipScale;

            inputs.push_back( FootprintInputs( x, y, 0, dPdx_x, dPdx_y, dPdy_x, dPdy_y ) );
        }
    }

    // Run test.
    std::vector<TileCoords> expectedTileCoords{// Fine level
                                               {{0, 1, {{0, 0}}},
                                                {0, 1, {{1, 0}}},
                                                {0, 1, {{2, 0}}},
                                                {0, 1, {{3, 0}}},
                                                {0, 1, {{4, 0}}},
                                                {0, 1, {{5, 0}}},
                                                {1, 2, {{2, 0}, {3, 0}}},
                                                {2, 1, {{1, 0}}},
                                                {3, 2, {{0, 0}, {1, 0}}},
                                                {3, 1, {{1, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {0, 1, {{0, 1}}},
                                                {0, 1, {{1, 1}}},
                                                {0, 1, {{2, 1}}},
                                                {0, 1, {{3, 1}}},
                                                {0, 1, {{4, 1}}},
                                                {0, 1, {{5, 1}}},
                                                {1, 2, {{2, 0}, {3, 0}}},
                                                {2, 1, {{1, 0}}},
                                                {3, 2, {{0, 0}, {1, 0}}},
                                                {3, 1, {{1, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {0, 1, {{0, 2}}},
                                                {0, 1, {{1, 2}}},
                                                {0, 1, {{2, 2}}},
                                                {0, 1, {{3, 2}}},
                                                {0, 1, {{4, 2}}},
                                                {0, 1, {{5, 2}}},
                                                {1, 2, {{2, 1}, {3, 1}}},
                                                {2, 1, {{1, 0}}},
                                                {3, 2, {{0, 0}, {1, 0}}},
                                                {3, 1, {{1, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {0, 1, {{0, 3}}},
                                                {0, 1, {{1, 3}}},
                                                {0, 1, {{2, 3}}},
                                                {0, 1, {{3, 3}}},
                                                {0, 1, {{4, 3}}},
                                                {0, 1, {{5, 3}}},
                                                {1, 2, {{2, 1}, {3, 1}}},
                                                {2, 1, {{1, 0}}},
                                                {3, 2, {{0, 0}, {1, 0}}},
                                                {3, 1, {{1, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {0, 1, {{0, 4}}},
                                                {0, 1, {{1, 4}}},
                                                {0, 1, {{2, 4}}},
                                                {0, 1, {{3, 4}}},
                                                {2, 1, {{1, 1}}},
                                                {2, 1, {{1, 1}}},
                                                {2, 1, {{1, 1}}},
                                                {2, 1, {{1, 1}}},
                                                {3, 2, {{0, 0}, {1, 0}}},
                                                {3, 1, {{1, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {0, 1, {{0, 5}}},
                                                {0, 1, {{1, 5}}},
                                                {0, 1, {{2, 5}}},
                                                {0, 1, {{3, 5}}},
                                                {2, 1, {{1, 1}}},
                                                {3, 1, {{0, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {3, 2, {{0, 0}, {1, 0}}},
                                                {3, 1, {{1, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {1, 2, {{0, 2}, {0, 3}}},
                                                {1, 2, {{0, 2}, {0, 3}}},
                                                {1, 2, {{1, 2}, {1, 3}}},
                                                {1, 2, {{1, 2}, {1, 3}}},
                                                {2, 1, {{1, 1}}},
                                                {3, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {2, 1, {{0, 1}}},
                                                {2, 1, {{0, 1}}},
                                                {2, 1, {{0, 1}}},
                                                {2, 1, {{0, 1}}},
                                                {2, 1, {{1, 1}}},
                                                {3, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {3, 2, {{0, 0}, {0, 1}}},
                                                {3, 2, {{0, 0}, {0, 1}}},
                                                {3, 2, {{0, 0}, {0, 1}}},
                                                {3, 2, {{0, 0}, {0, 1}}},
                                                {3, 2, {{0, 0}, {0, 1}}},
                                                {3, 2, {{0, 0}, {0, 1}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {3, 1, {{0, 1}}},
                                                {3, 1, {{0, 1}}},
                                                {3, 1, {{0, 1}}},
                                                {3, 1, {{0, 1}}},
                                                {3, 1, {{0, 1}}},
                                                {3, 1, {{0, 1}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},  // End fine level
                                                {0, 0, {}},        // Coarse level
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {1, 1, {{2, 0}}},
                                                {2, 1, {{1, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {1, 1, {{2, 0}}},
                                                {2, 1, {{1, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {1, 1, {{2, 1}}},
                                                {2, 1, {{1, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {1, 1, {{1, 1}}},
                                                {1, 1, {{2, 1}}},
                                                {1, 1, {{2, 1}}},
                                                {2, 1, {{1, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {0, 0, {}},
                                                {1, 1, {{1, 2}}},
                                                {3, 1, {{0, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {1, 1, {{0, 2}}},
                                                {1, 1, {{0, 2}}},
                                                {1, 1, {{1, 2}}},
                                                {1, 1, {{1, 2}}},
                                                {3, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {2, 1, {{0, 1}}},
                                                {2, 1, {{0, 1}}},
                                                {2, 1, {{0, 1}}},
                                                {2, 1, {{0, 1}}},
                                                {3, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {3, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {4, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {5, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {6, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {10, 1, {{0, 0}}},
                                                {10, 1, {{0, 0}}},
                                                {10, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {10, 1, {{0, 0}}},
                                                {10, 1, {{0, 0}}},
                                                {10, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {7, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {8, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {9, 1, {{0, 0}}},
                                                {10, 1, {{0, 0}}},
                                                {10, 1, {{0, 0}}},
                                                {10, 1, {{0, 0}}}}};

    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprintGrad", inputs, expectedTileCoords, cudaAddressModeClamp ) );
}

TEST_F( TextureFootprintTest, TestRequestMultipleLevels )
{
    std::vector<FootprintInputs> inputs;
    inputs.push_back( FootprintInputs( 0.5, 0.5, 1.5 ) );
    inputs.push_back( FootprintInputs( 0.25, 0.25, 2.5 ) );
    inputs.push_back( FootprintInputs( 0.0f, 0.0f, 3.5f ) );
    inputs.push_back( FootprintInputs( 0.5f, 0.5f, 4.5f ) );

    // Run test.
    std::vector<TileCoords> expectedTileCoords{// Fine level
                                               {1, 4, {{3, 3}, {4, 3}, {3, 4}, {4, 4}}},
                                               {2, 4, {{0, 0}, {1, 0}, {0, 1}, {1, 1}}},
                                               {3, 1, {{0, 0}}},
                                               {4, 1, {{0, 0}}},
                                               // Coarse level
                                               {2, 4, {{1, 1}, {2, 1}, {1, 2}, {2, 2}}},
                                               {3, 1, {{0, 0}}},
                                               {4, 1, {{0, 0}}},
                                               {5, 1, {{0, 0}}}};
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprintLod", inputs, expectedTileCoords, cudaAddressModeClamp ) );
}

TEST_F( TextureFootprintTest, TestMipTail )
{
    std::vector<FootprintInputs> inputs;
    inputs.push_back( FootprintInputs( 0.5, 0.5, 3.5 ) );  // Straddle mip tail and tiles
    inputs.push_back( FootprintInputs( 0.5, 0.5, 4.5 ) );  // Just inside mip tail
    inputs.push_back( FootprintInputs( 0.5, 0.5, 20.5f ) );  // Higher than the highest level

    // Run test.
    std::vector<TileCoords> expectedTileCoords{{3, 4, {{0, 0}, {1, 0}, {0, 1}, {1, 1}}},
                                               {4, 1, {{0, 0}}},
                                               {10, 1, {{0, 0}}},
                                               {4, 1, {{0, 0}}},
                                               {5, 1, {{0, 0}}},
                                               {0, 0, {}}};
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprintLod", inputs, expectedTileCoords, cudaAddressModeClamp ) );
}

TEST_F( TextureFootprintTest, TestWrapMode )
{
    // Corners, edges, and outside [0,1]
    std::vector<FootprintInputs> inputs{FootprintInputs( 0.f, 0.f, 0.0f ),     FootprintInputs( 1.f, 0.f, 0.0f ),
                                        FootprintInputs( 0.f, 1.f, 0.0f ),     FootprintInputs( 1.f, 1.f, 0.0f ),
                                        FootprintInputs( 0.25f, 0.0f, 0.0f ),  FootprintInputs( 0.25f, 1.0f, 0.0f ),
                                        FootprintInputs( 0.0f, 0.25f, 0.0f ),  FootprintInputs( 1.0f, 0.25f, 0.0f ),
                                        FootprintInputs( -0.5f, -0.5f, 0.0f ), FootprintInputs( 1.5f, 1.5f, 0.0f ),
                                        FootprintInputs( -3.5f, -3.5f, 0.0f ), FootprintInputs( 10.5f, 3.5f, 0.0f ),
                                        FootprintInputs( -0.1f, 0.1f, 0.0f ),  FootprintInputs( 0.f, 0.f, 1.0f ),
                                        FootprintInputs( 1.f, 0.f, 1.0f ),     FootprintInputs( 0.f, 1.f, 1.0f ),
                                        FootprintInputs( 1.f, 1.f, 1.0f ),     FootprintInputs( 0.25f, 0.0f, 1.0f ),
                                        FootprintInputs( 0.25f, 1.0f, 1.0f ),  FootprintInputs( 0.0f, 0.25f, 1.0f ),
                                        FootprintInputs( 1.0f, 0.25f, 1.0f ),  FootprintInputs( -0.5f, -0.5f, 1.0f ),
                                        FootprintInputs( 1.5f, 1.5f, 1.0f ),   FootprintInputs( -3.5f, -3.5f, 1.0f ),
                                        FootprintInputs( 10.5f, 3.5f, 1.0f ),  FootprintInputs( 0.0f, 0.1f, 1.0f )};

    // Run test
    std::vector<TileCoords> expectedTileCoords{{0, 4, {{0, 0}, {15, 0}, {0, 15}, {15, 15}}},
                                               {0, 4, {{0, 0}, {15, 0}, {0, 15}, {15, 15}}},
                                               {0, 4, {{0, 0}, {15, 0}, {0, 15}, {15, 15}}},
                                               {0, 4, {{0, 0}, {15, 0}, {0, 15}, {15, 15}}},
                                               {0, 4, {{3, 0}, {4, 0}, {3, 15}, {4, 15}}},
                                               {0, 4, {{3, 0}, {4, 0}, {3, 15}, {4, 15}}},
                                               {0, 4, {{0, 3}, {15, 3}, {0, 4}, {15, 4}}},
                                               {0, 4, {{0, 3}, {15, 3}, {0, 4}, {15, 4}}},
                                               {0, 4, {{8, 8}, {7, 8}, {8, 7}, {7, 7}}},
                                               {0, 4, {{8, 8}, {7, 8}, {8, 7}, {7, 7}}},
                                               {0, 4, {{8, 8}, {7, 8}, {8, 7}, {7, 7}}},
                                               {0, 4, {{8, 8}, {7, 8}, {8, 7}, {7, 7}}},
                                               {0, 1, {{14, 1}}},  //
                                               {1, 4, {{0, 0}, {7, 0}, {0, 7}, {7, 7}}},
                                               {1, 4, {{0, 0}, {7, 0}, {0, 7}, {7, 7}}},
                                               {1, 4, {{0, 0}, {7, 0}, {0, 7}, {7, 7}}},
                                               {1, 4, {{0, 0}, {7, 0}, {0, 7}, {7, 7}}},
                                               {1, 4, {{1, 0}, {2, 0}, {1, 7}, {2, 7}}},
                                               {1, 4, {{1, 0}, {2, 0}, {1, 7}, {2, 7}}},
                                               {1, 4, {{0, 1}, {7, 1}, {0, 2}, {7, 2}}},
                                               {1, 4, {{0, 1}, {7, 1}, {0, 2}, {7, 2}}},
                                               {1, 4, {{3, 3}, {4, 3}, {3, 4}, {4, 4}}},
                                               {1, 4, {{3, 3}, {4, 3}, {3, 4}, {4, 4}}},
                                               {1, 4, {{3, 3}, {4, 3}, {3, 4}, {4, 4}}},
                                               {1, 4, {{3, 3}, {4, 3}, {3, 4}, {4, 4}}},
                                               {1, 2, {{0, 0}, {7, 0}}}};

    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprintLod", inputs, expectedTileCoords, cudaAddressModeWrap ) );
}

TEST_F( TextureFootprintTest, TestOddsizeWrap )
{
    // Gradients that wrap
    std::vector<FootprintInputs> inputs;
    const float                  lod = 3.1f;
    inputs.push_back( FootprintInputs( 0.99999f, 0.1f, lod, 0.0055f, 0.0f, 0.0f, 0.0055f ) );  // left edge
    inputs.push_back( FootprintInputs( 0.00001f, 0.1f, lod, 0.0055f, 0.0f, 0.0f, 0.0055f ) );  // right edge
    inputs.push_back( FootprintInputs( 0.77599f, 0.00001f, 1.1f, -0.00238f, -0.00150f, -0.00034f, 0.00152f ) );

    // Run test.
    m_fixture->setTextureWidth( 2000 );
    m_fixture->setTextureHeight( 1500 );

    // Note: The comment below shows the coordinates that *should* be produced by this test.
    // However, only the footprint near the left edge of the texture (at x=1.0f-0.99999f) wraps properly.
    // The footprint near the right of the texture (x=0.99999f) misses the wrapped tile.
    // This is likely due to the edge of the texture being in the middle
    // of a texel group for the given granularity.
    // std::vector<TileCoords> expectedTileCoords{{ 3, 2, {{0, 0}, {3, 0}}}, { 3, 2, {{0, 0}, {3, 0}}}, ...};

    std::vector<TileCoords> expectedTileCoords{{3, 1, {{3, 0}}},  //
                                               {3, 2, {{0, 0}, {3, 0}}},
                                               {1, 2, {{12, 0}, {12, 11}}}};
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprintGrad", inputs, expectedTileCoords, cudaAddressModeWrap ) );
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprintLod", inputs, expectedTileCoords, cudaAddressModeWrap ) );
}

TEST_F( TextureFootprintTest, TestOddsizeWrapX )
{
    // Gradients that wrap
    std::vector<FootprintInputs> inputs;
    inputs.push_back( FootprintInputs( 0.00099f, 0.77599f, 1.1f, -0.00150f, -0.00238f, 0.00152f, -0.00034f ) );

    // Run test.
    m_fixture->setTextureWidth( 1500 );
    m_fixture->setTextureHeight( 2000 );

    std::vector<TileCoords> expectedTileCoords{{1, 2, {{0, 12}, {11, 12}}}};
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprintGrad", inputs, expectedTileCoords, cudaAddressModeWrap ) );
}

TEST_F( TextureFootprintTest, TestMoreOddSizeWrap )
{
    // Gradients that wrap
    std::vector<FootprintInputs> inputs;
    inputs.push_back( FootprintInputs( 0.9974206686f, 0.9383302331f, 1.0f, 2.777942973e-05f, -0.0005159105058f,
                                       0.003821033984f, -0.006097915582f ) );
    inputs.push_back( FootprintInputs( 0.0f, 0.9383302331f, 1.0f, 2.777942973e-05f, -0.0005159105058f, 0.003821033984f, -0.006097915582f ) );

    // Run test.
    m_fixture->setTextureWidth( 4095 );
    m_fixture->setTextureHeight( 4095 );

    // Note: The expectedTileCoordinates below show the correction that is made in Texture2D.h to account for
    // wrapping, which in this case moves the x coordinate to 0. The expected coordinates demonstrate that the
    // tiles from the original are not always both present at the changed location (where x=0).

    std::vector<TileCoords> expectedTileCoords{{0, 2, {{63, 59}, {63, 60}}},
                                               {0, 3, {{0, 59}, {0, 60}, {63, 60}}},
                                               {1, 2, {{31, 29}, {31, 30}}},
                                               {1, 3, {{0, 29}, {0, 30}, {31, 30}}}};
    EXPECT_EQ( 0, m_fixture->runTest( "__raygen__testFootprintGrad", inputs, expectedTileCoords, cudaAddressModeWrap ) );
}
