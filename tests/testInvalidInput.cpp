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

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>
#include <OptiXToolkit/Util/Exception.h>
#include <OptiXToolkit/Util/Files.h>
#include <OptiXToolkit/Util/CuBuffer.h>

#include "Util/Image.h"
#include "Util/Mesh.h"
#include "Util/OptiXOmmArray.h"
#include "Util/OptiXScene.h"
#include "Util/BakeTexture.h"

#include "testCommon.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {  // anonymous

    // Check status returned by a CUDA call.
    inline void check( cudaError_t status )
    {
        if( status != cudaSuccess )
            throw std::runtime_error( cudaGetErrorString( status ) );
    }

    // Check status returned by a CuOmmBaking call.
    inline void check( cuOmmBaking::Result status )
    {
        if( status != cuOmmBaking::Result::SUCCESS )
            throw std::runtime_error( "Omm baking failure." );
    }

    const uint3 g_indices[2] = {
        {0,1,2},
        {1,3,2}
    };

    const float2 g_texCoords[6] =
    {
        0.f, 0.f,
        1.f, 0.f,
        0.f, 1.f,
        1.f, 0.f,
        1.f, 1.f,
        0.f, 1.f,
    };

    const float3 g_vertices[4] =
    {
        0.f, 0.f, 0.f,
        1.f, 0.f, 0.f,
        0.f, 1.f, 0.f,
        1.f, 1.f, 0.f,
    };

    const uint32_t g_width = 4;
    const uint32_t g_height = 4;

    const uint8_t g_states[g_width * g_height] =
    {
        0, 0, 1, 1,
        0, 0, 1, 1,
        1, 1, 0, 0,
        1, 1, 0, 0
    };

    const float2 g_transform[3] = { {1,0}, {0,1}, {0,0} };

    const uint32_t g_texIndices[2] = { 0, 1 };

    class InvalidInputTests : public ::testing::TestWithParam <unsigned int>
    {
    public:
        const static unsigned int NUM_TEST_CASES = 87;
    protected:

        cuOmmBaking::Result m_expectedResult = cuOmmBaking::Result::SUCCESS;

        struct TestOptions
        {
            unsigned int testCase = 0;
        };

        cudaMipmappedArray_t m_mipmap = {};

        void SetUp() override
        {
            // Initialize CUDA runtime
            CUDA_THROW( cudaFree( 0 ) );
        }

        void TearDown() override
        {
            check( cudaFreeMipmappedArray( m_mipmap ) );
        }

        cuOmmBaking::Result runTest( const TestOptions& opt )
        {
            cuOmmBaking::Result result;

            cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
            cudaExtent extent = { 128,128,1 };
            bool layered = false;
            uint32_t mipmaps = 1;

            if( opt.testCase == 50 )
            {
                desc = cudaCreateChannelDesc<uint2>();
            }

            cuOmmBaking::CudaTextureAlphaMode alphaMode = cuOmmBaking::CudaTextureAlphaMode::DEFAULT;

            if( opt.testCase == 82 )
            {
                alphaMode = cuOmmBaking::CudaTextureAlphaMode::MAX_NUM;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 83 )
            {
                desc = cudaCreateChannelDesc<uint2>();
                alphaMode = cuOmmBaking::CudaTextureAlphaMode::CHANNEL_W;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 84 )
            {
                desc = cudaCreateChannelDesc<uint2>();
                alphaMode = cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Z;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 85 )
            {
                desc = cudaCreateChannelDesc<uint2>();
                alphaMode = cuOmmBaking::CudaTextureAlphaMode::RGB_INTENSITY;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 86 )
            {
                desc = cudaCreateChannelDesc<unsigned int>();
                alphaMode = cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Y;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }

            check( cudaMallocMipmappedArray( &m_mipmap, &desc, extent, mipmaps, layered ? cudaArrayLayered : 0 ) );

            struct cudaResourceDesc resDesc;
            memset( &resDesc, 0, sizeof( resDesc ) );
            resDesc.resType = cudaResourceTypeMipmappedArray;
            resDesc.res.mipmap.mipmap = m_mipmap;

            struct cudaTextureDesc texDesc;
            memset( &texDesc, 0, sizeof( texDesc ) );
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 1;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.maxMipmapLevelClamp = ( float )( mipmaps - 1 );
            texDesc.mipmapFilterMode = cudaFilterModePoint;

            cudaTextureObject_t  tex;
            check( cudaCreateTextureObject( &tex, &resDesc, &texDesc, 0 ) );

            // Upload geometry data

            CuBuffer<uint3> devIndices;
            CuBuffer<float2> devTexCoords;
            CuBuffer<uint8_t> devStates;
            CuBuffer<uint32_t> devTexIndices;
            CuBuffer<float2> devTransform;

            devIndices.allocAndUpload( sizeof( g_indices ) / sizeof(uint3), g_indices);
            devTexCoords.allocAndUpload( sizeof( g_texCoords ) / sizeof( uint3 ), g_texCoords );
            devStates.allocAndUpload( sizeof( g_states ) / sizeof( uint8_t ), g_states );
            devTexIndices.allocAndUpload( sizeof( g_texIndices ) / sizeof( uint32_t ), g_texIndices );
            devTransform.allocAndUpload( sizeof( g_transform ) / sizeof( float2 ), g_transform );
            
            cuOmmBaking::BakeOptions ommOptions = {};
            ommOptions.flags = cuOmmBaking::BakeFlags::ENABLE_POST_BAKE_INFO;

            cuOmmBaking::TextureDesc texture[2] = { {}, {} };
            texture[0].type = cuOmmBaking::TextureType::CUDA;
            texture[0].cuda.texObject = tex;
            texture[0].cuda.transparencyCutoff = 0.f;
            texture[0].cuda.opacityCutoff = 1.f;
            texture[0].cuda.alphaMode = alphaMode;

            texture[1].type = cuOmmBaking::TextureType::STATE;
            texture[1].state.width = g_width;
            texture[1].state.height = g_height;

            cuOmmBaking::BakeInputDesc input = {};
            input.indexFormat = cuOmmBaking::IndexFormat::I32_UINT;
            input.numIndexTriplets = 2;

            input.texCoordFormat = cuOmmBaking::TexCoordFormat::UV32_FLOAT2;
            input.texCoordBuffer = 0;
            input.numTexCoords = 6;

            input.numTextures = 2;
            input.textures = texture;

            input.textureIndexFormat = cuOmmBaking::IndexFormat::I32_UINT;
            
            input.transformFormat = cuOmmBaking::UVTransformFormat::MATRIX_FLOAT2X3;

            if( opt.testCase == 1 )
            {
                input.indexFormat = cuOmmBaking::IndexFormat::NONE;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 2 )
            {
                input.indexFormat = cuOmmBaking::IndexFormat::NONE;
                input.numIndexTriplets = 0;
                input.numTexCoords = 4;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 3 )
            {
                input.indexFormat = cuOmmBaking::IndexFormat::I8_UINT;
                input.numTexCoords = 1 << 8;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 4 )
            {
                input.indexFormat = cuOmmBaking::IndexFormat::I16_UINT;
                input.numTexCoords = 1 << 16;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 5 )
            {
                input.indexFormat = cuOmmBaking::IndexFormat::MAX_NUM;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 6 )
            {
                input.indexFormat = cuOmmBaking::IndexFormat::NONE;
                input.numIndexTriplets = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 7 )
            {
                input.numIndexTriplets = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 8 )
            {
                input.indexFormat = cuOmmBaking::IndexFormat::NONE;
                input.numIndexTriplets = 0;
                input.numTexCoords = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 9 )
            {
                input.texCoordFormat = cuOmmBaking::TexCoordFormat::MAX_NUM;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 10 )
            {
                input.numTextures = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 11 )
            {
                input.textures = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 12 )
            {
                input.numTextures = 1;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 13 )
            {
                input.textureIndexFormat = cuOmmBaking::IndexFormat::I8_UINT;
                input.numTextures = 1 << 8;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 14 )
            {
                input.textureIndexFormat = cuOmmBaking::IndexFormat::I16_UINT;
                input.numTextures = 1 << 16;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 15 )
            {
                input.textureIndexFormat = cuOmmBaking::IndexFormat::MAX_NUM;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 16 )
            {
                input.textureIndexFormat = cuOmmBaking::IndexFormat::NONE;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 17 )
            {
                input.texCoordStrideInBytes = 6;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 18 )
            {
                input.indexFormat = cuOmmBaking::IndexFormat::I16_UINT;
                input.indexTripletStrideInBytes = 3;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 19 )
            {
                input.textureIndexStrideInBytes = 6;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 20 )
            {
                input.textureIndexFormat = cuOmmBaking::IndexFormat::I16_UINT;
                input.textureIndexStrideInBytes = 1;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 21 )
            {
                input.textureIndexStrideInBytes = 2;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 22 )
            {
                input.transformFormat = cuOmmBaking::UVTransformFormat::MAX_NUM;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }

            if( opt.testCase == 35 )
            {
                texture[1].state.pitchInBits = 1;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 36 )
            {
                texture[1].state.width = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 37 )
            {
                texture[1].state.height = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 38 )
            {
                texture[1].state.width = 1 << 15;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 39 )
            {
                texture[1].state.height = 1 << 15;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 40)
            {
                texture[1].state.filterKernelWidthInTexels = -1;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 41 )
            {
                texture[1].state.filterKernelWidthInTexels = std::nanf( "1" );
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 42 )
            {
                texture[1].state.addressMode[0] = cudaAddressModeBorder;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 43 )
            {
                texture[1].state.addressMode[1] = cudaAddressModeBorder;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            
            if( opt.testCase == 44 )
            {
                texture[0].cuda.texObject = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 45 )
            {
                texture[0].cuda.transparencyCutoff = 2.f;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 46 )
            {
                texture[0].cuda.opacityCutoff = -1.f;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 47 )
            {
                texture[0].cuda.filterKernelWidthInTexels = -1;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 48 )
            {
                texture[0].cuda.filterKernelWidthInTexels = std::nanf( "1" );
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 49 )
            {
                texture[1].type = cuOmmBaking::TextureType::MAX_NUM;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }

            if( opt.testCase == 51 )
            {
                ommOptions.flags = ( cuOmmBaking::BakeFlags )~0u;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }

            cuOmmBaking::BakeInputBuffers outInputBuffer;
            cuOmmBaking::BakeBuffers outBuffers;

            cuOmmBaking::BakeOptions* argOptions = &ommOptions;
            cuOmmBaking::BakeInputDesc* argInput = &input;
            cuOmmBaking::BakeInputBuffers* argOutInputBuffer = &outInputBuffer;
            cuOmmBaking::BakeBuffers* argOutBuffers = &outBuffers;
            uint32_t numInputs = 1;

            if( opt.testCase == 75 )
            {
                numInputs = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 76 )
            {
                argOptions = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 77 )
            {
                argInput = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 78 )
            {
                argOutInputBuffer = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 79 )
            {
                argOutBuffers = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            
            if( opt.testCase == 80 )
            {
                input.numIndexTriplets = 1 << 29;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 81 )
            {
                input.indexFormat = cuOmmBaking::IndexFormat::NONE;
                input.numIndexTriplets = 0;
                input.numTexCoords = ( 1 << 29 ) * 3;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }

            // Prepare for baking by query the pre baking info
            
            if( ( result = cuOmmBaking::GetPreBakeInfo( argOptions, numInputs, argInput, argOutInputBuffer, argOutBuffers ) ) != cuOmmBaking::Result::SUCCESS )
                return result;

            if( opt.testCase == 74 )
            {
                ommOptions.flags = cuOmmBaking::BakeFlags::NONE;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }

            // Allocate baking output buffers

            CuBuffer <> indexBuffer;
            CuBuffer <OptixOpacityMicromapUsageCount> micromapUsageCountsBuffer;
            CuBuffer <> outputBuffer;
            CuBuffer <OptixOpacityMicromapDesc> perMicromapDescBuffer;
            CuBuffer <OptixOpacityMicromapHistogramEntry> micromapHistogramEntriesBuffer;
            CuBuffer <> postBakeInfoBuffer;
            CuBuffer <> temp;

            indexBuffer.alloc( outInputBuffer.indexBufferSizeInBytes );
            micromapUsageCountsBuffer.alloc( outInputBuffer.numMicromapUsageCounts );
            outputBuffer.alloc( outBuffers.outputBufferSizeInBytes );
            perMicromapDescBuffer.alloc( outBuffers.numMicromapDescs );
            micromapHistogramEntriesBuffer.alloc( outBuffers.numMicromapHistogramEntries );
            postBakeInfoBuffer.alloc( outBuffers.postBakeInfoBufferSizeInBytes );
            temp.alloc( outBuffers.tempBufferSizeInBytes );

            outInputBuffer.indexBuffer = indexBuffer.get();
            outInputBuffer.micromapUsageCountsBuffer = micromapUsageCountsBuffer.get();
            outBuffers.outputBuffer = outputBuffer.get();
            outBuffers.perMicromapDescBuffer = perMicromapDescBuffer.get();
            outBuffers.micromapHistogramEntriesBuffer = micromapHistogramEntriesBuffer.get();
            outBuffers.postBakeInfoBuffer = postBakeInfoBuffer.get();
            outBuffers.tempBuffer = temp.get();

            input.indexBuffer = devIndices.get();
            input.texCoordBuffer = devTexCoords.get();
            input.textureIndexBuffer = devTexIndices.get();
            input.transform = devTransform.get();
            texture[1].state.stateBuffer = devStates.get();

            if( opt.testCase == 23 )
            { 
                input.indexBuffer = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 24 )
            {
                input.indexFormat = cuOmmBaking::IndexFormat::I16_UINT;
                input.indexBuffer = 1;
                m_expectedResult = cuOmmBaking::Result::ERROR_MISALIGNED_ADDRESS;
            }
            else if( opt.testCase == 25 )
            {
                input.indexFormat = cuOmmBaking::IndexFormat::I32_UINT;
                input.indexBuffer = 2;
                m_expectedResult = cuOmmBaking::Result::ERROR_MISALIGNED_ADDRESS;
            }
            else if( opt.testCase == 26 )
            {
                input.indexFormat = cuOmmBaking::IndexFormat::NONE;
                input.indexBuffer = 2;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 27 )
            {
                input.textureIndexBuffer = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 28 )
            {
                input.textureIndexFormat = cuOmmBaking::IndexFormat::I16_UINT;
                input.textureIndexBuffer = 1;
                m_expectedResult = cuOmmBaking::Result::ERROR_MISALIGNED_ADDRESS;
            }
            else if( opt.testCase == 29 )
            {
                input.textureIndexFormat = cuOmmBaking::IndexFormat::I32_UINT;
                input.textureIndexBuffer = 2;
                m_expectedResult = cuOmmBaking::Result::ERROR_MISALIGNED_ADDRESS;
            }
            else if( opt.testCase == 30 )
            {
                input.textureIndexFormat = cuOmmBaking::IndexFormat::NONE;
                input.textureIndexBuffer = 2;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 31 )
            {
                input.texCoordBuffer = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 32 )
            {
                input.transform = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 33 )
            {
                input.transform = 1;
                m_expectedResult = cuOmmBaking::Result::ERROR_MISALIGNED_ADDRESS;
            }
            else if( opt.testCase == 34 )
            {
                texture[1].state.stateBuffer = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }

            if( opt.testCase == 52 )
            {
                outBuffers.outputBuffer = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 53 )
            {
                outBuffers.outputBuffer = 1;
                m_expectedResult = cuOmmBaking::Result::ERROR_MISALIGNED_ADDRESS;
            }
            else if( opt.testCase == 54 )
            {
                outBuffers.outputBufferSizeInBytes = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 55 )
            {
                outBuffers.perMicromapDescBuffer = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 56 )
            {
                outBuffers.perMicromapDescBuffer = 1;
                m_expectedResult = cuOmmBaking::Result::ERROR_MISALIGNED_ADDRESS;
            }
            else if( opt.testCase == 57 )
            {
                outBuffers.numMicromapDescs = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }

            else if( opt.testCase == 58 )
            {
                outBuffers.micromapHistogramEntriesBuffer = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 59 )
            {
                outBuffers.micromapHistogramEntriesBuffer = 1;
                m_expectedResult = cuOmmBaking::Result::ERROR_MISALIGNED_ADDRESS;
            }
            else if( opt.testCase == 60 )
            {
                outBuffers.numMicromapHistogramEntries = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }

            else if( opt.testCase == 61 )
            {
                outBuffers.postBakeInfoBuffer = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 62 )
            {
                outBuffers.postBakeInfoBuffer = 1;
                m_expectedResult = cuOmmBaking::Result::ERROR_MISALIGNED_ADDRESS;
            }
            else if( opt.testCase == 63 )
            {
                outBuffers.postBakeInfoBufferSizeInBytes = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }

            else if( opt.testCase == 64 )
            {
                outBuffers.tempBuffer = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 65 )
            {
                outBuffers.tempBufferSizeInBytes = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }

            else if( opt.testCase == 66 )
            {
                outInputBuffer.indexBuffer = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 67 )
            {
                outInputBuffer.indexBuffer = 1;
                m_expectedResult = cuOmmBaking::Result::ERROR_MISALIGNED_ADDRESS;
            }

            else if( opt.testCase == 68 )
            {
                outInputBuffer.micromapUsageCountsBuffer = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 69 )
            {
                outInputBuffer.micromapUsageCountsBuffer = 1;
                m_expectedResult = cuOmmBaking::Result::ERROR_MISALIGNED_ADDRESS;
            }
            else if( opt.testCase == 70 )
            {
                outInputBuffer.numMicromapUsageCounts = 0;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }

            if( opt.testCase == 71 )
            {
                outBuffers.indexFormat = cuOmmBaking::IndexFormat::NONE;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 72 )
            {
                outBuffers.indexFormat = cuOmmBaking::IndexFormat::MAX_NUM;
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }
            else if( opt.testCase == 73 )
            {
                switch( outBuffers.indexFormat )
                {
                case cuOmmBaking::IndexFormat::I8_UINT:
                    outBuffers.indexFormat = cuOmmBaking::IndexFormat::I16_UINT;
                    break;
                case cuOmmBaking::IndexFormat::I16_UINT:
                    outBuffers.indexFormat = cuOmmBaking::IndexFormat::I32_UINT;
                    break;
                case cuOmmBaking::IndexFormat::I32_UINT:
                    outBuffers.indexFormat = cuOmmBaking::IndexFormat::I8_UINT;
                    break;
                }
                m_expectedResult = cuOmmBaking::Result::ERROR_INVALID_VALUE;
            }

            // Execute the baking

            if( ( result = cuOmmBaking::BakeOpacityMicromaps( argOptions, numInputs, argInput, argOutInputBuffer, argOutBuffers, 0 ) ) != cuOmmBaking::Result::SUCCESS )
                return result;

            check( cudaDeviceSynchronize() );

            return cuOmmBaking::Result::SUCCESS;
        }

    };
}

TEST_P( InvalidInputTests, Base )
{
    TestOptions opt = {};
    opt.testCase = GetParam();

    cuOmmBaking::Result res = runTest( opt );
    ASSERT_EQ( res, m_expectedResult );
}

INSTANTIATE_TEST_SUITE_P(
    Base,
    InvalidInputTests,
    ::testing::Range(0u, InvalidInputTests::NUM_TEST_CASES));