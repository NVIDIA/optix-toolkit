// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <testCuOmmBakingKernelsCuda.h>

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <optix_types.h>

#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#include <OptiXToolkit/OptiXMemory/CompileOptions.h>

#include "OptiXScene.h"

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif

#ifndef NDEBUG

#define CUDA_CHECK( call )                                                                                                                                                                                                                                                             \
    {                                                                                                                                                                                                                                                                                  \
        cudaError_t error = call;                                                                                                                                                                                                                                                      \
        if( error == cudaSuccess )                                                                                                                                                                                                                                                     \
            error = cudaDeviceSynchronize();                                                                                                                                                                                                                                           \
        if( error != cudaSuccess )                                                                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                                                                              \
            return OPTIX_ERROR_CUDA_ERROR;                                                                                                                                                                                                                                             \
        }                                                                                                                                                                                                                                                                              \
    }

#define OPTIX_CHECK( call )                                                                                                                                                                                                                                                            \
    {                                                                                                                                                                                                                                                                                  \
        OptixResult res = call;                                                                                                                                                                                                                                                        \
        if( res != OPTIX_SUCCESS )                                                                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                                                                              \
            return res;                                                                                                                                                                                                                                                                \
        }                                                                                                                                                                                                                                                                              \
        cudaError_t error = cudaDeviceSynchronize();                                                                                                                                                                                                                                   \
        if( error != cudaSuccess )                                                                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                                                                              \
            return OPTIX_ERROR_CUDA_ERROR;                                                                                                                                                                                                                                             \
        }                                                                                                                                                                                                                                                                              \
    }

#else

#define OPTIX_CHECK( call )                                                                                                                                                                                                                                                            \
    {                                                                                                                                                                                                                                                                                  \
        OptixResult res = call;                                                                                                                                                                                                                                                        \
        if( res != OPTIX_SUCCESS )                                                                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                                                                              \
            return res;                                                                                                                                                                                                                                                                \
        }                                                                                                                                                                                                                                                                              \
    }

#define CUDA_CHECK( call )                                                                                                                                                                                                                                                             \
    {                                                                                                                                                                                                                                                                                  \
        cudaError_t error = call;                                                                                                                                                                                                                                                      \
        if( error != cudaSuccess )                                                                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                                                                              \
            return OPTIX_ERROR_CUDA_ERROR;                                                                                                                                                                                                                                             \
        }                                                                                                                                                                                                                                                                              \
    }

#endif

#define CUDA_SYNC_CHECK() CUDA_CHECK( cudaDeviceSynchronize() )

uint32_t getIndexFormatSizeInBytes( cuOmmBaking::IndexFormat format )
{
    switch( format )
    {
    case cuOmmBaking::IndexFormat::NONE:
        return 0;
    case cuOmmBaking::IndexFormat::I8_UINT:
        return 1;
    case cuOmmBaking::IndexFormat::I16_UINT:
        return 2;
    case cuOmmBaking::IndexFormat::I32_UINT:
        return 4;
    default:
        return 0;
    }
}

OptixResult OptixOmmScene::build( OptixDeviceContext context, const char* optixirInput, const size_t optixirInputize, const OptixOmmArray& optixOmm, const cuOmmBaking::BakeInputDesc* ommBuildInput, unsigned int numBuildInputs )
{
    OPTIX_CHECK( destroy() );

    m_context = context;

    CUDA_SYNC_CHECK();

    if( OptixResult result = buildPipeline( optixirInput, optixirInputize ) )
    {
        destroy();
        OPTIX_CHECK( result );
    }

    CUDA_SYNC_CHECK();

    if( OptixResult result = buildGAS( optixOmm, ommBuildInput, numBuildInputs ) )
    {
        destroy();
        OPTIX_CHECK( result );
    }

    CUDA_SYNC_CHECK();

    if( OptixResult result = buildSBT( ommBuildInput, numBuildInputs ) )
    {
        destroy();
        OPTIX_CHECK( result );
    }

    return OPTIX_SUCCESS;
}

OptixResult OptixOmmScene::render( uint32_t width, uint32_t height, RenderOptions options )
{
    CUDA_CHECK( m_imageBuf.alloc( width * height ) );
    CUDA_CHECK( m_paramsBuf.alloc( 1 ) );
    CUDA_CHECK( m_errorCountBuf.alloc( 1 ) );

    Params params;
    params.image        = (uchar3*)m_imageBuf.get();
    params.image_width  = width;
    params.image_height = height;
    params.handle       = m_iasHandle;
    params.options      = options;
    params.error_count  = (uint32_t*)m_errorCountBuf.get();
    CUDA_CHECK( m_paramsBuf.upload( &params ) );

    m_errorCount = 0;
    m_errorCountBuf.upload( &m_errorCount );

    OPTIX_CHECK( optixLaunch( m_pipeline, 0, m_paramsBuf.get(), m_paramsBuf.byteSize(), &m_sbt, width, height, /*depth=*/1 ) );

    m_image.resize( width * height );
    CUDA_CHECK( m_imageBuf.download( m_image.data() ) );

    if( options.validate_opacity )
    {
        CUDA_CHECK( m_errorCountBuf.download( &m_errorCount ) );
    }

    return OPTIX_SUCCESS;
}

OptixResult OptixOmmScene::destroy()
{
    if( m_pipeline )
        optixPipelineDestroy( m_pipeline );
    if( m_rgProgramGroup )
        optixProgramGroupDestroy( m_rgProgramGroup );
    if( m_msProgramGroup[0] )
        optixProgramGroupDestroy( m_msProgramGroup[0] );
    if( m_msProgramGroup[1] )
        optixProgramGroupDestroy( m_msProgramGroup[1] );
    if( m_hitgroupProgramGroup[0] )
        optixProgramGroupDestroy( m_hitgroupProgramGroup[0] );
    if( m_hitgroupProgramGroup[1] )
        optixProgramGroupDestroy( m_hitgroupProgramGroup[1] );

    for( auto itr : m_dcProgramGroupMap )
        optixProgramGroupDestroy( itr.second );

    if( m_module )
        optixModuleDestroy( m_module );
    if( m_moduleDC )
        optixModuleDestroy( m_moduleDC );

    m_dcProgramGroupMap.clear();

    m_pipeline                = {};
    m_rgProgramGroup          = {};
    m_msProgramGroup[0]       = {};
    m_msProgramGroup[1]       = {};
    m_hitgroupProgramGroup[0] = {};
    m_hitgroupProgramGroup[1] = {};
    m_module                  = {};
    m_moduleDC                = {};
    m_context                 = {};

    m_image.clear();

    m_errorCount = {};

    m_sbt       = {};
    m_gasHandle = {};
    m_iasHandle = {};

    m_paramsBuf.free();
    m_imageBuf.free();
    m_errorCountBuf.free();
    m_gas.free();
    m_ias.free();
    m_hitSbtRecords.free();
    m_raygenSbtRecords.free();
    m_missSbtRecords.free();
    m_dcSbtRecords.free();

    return OPTIX_SUCCESS;
}

OptixResult OptixOmmScene::buildGAS( const OptixOmmArray& optixOmm, const cuOmmBaking::BakeInputDesc* ommBuildInput, unsigned int numBuildInputs )
{
    std::deque<unsigned int>                   flags;  // use deque so pointers don't get invalidated on push_back
    std::vector<OptixBuildInput>               bakeInputs;
    std::deque<OptixBuildInputOpacityMicromap> vms;
    for( unsigned int i = 0; i < numBuildInputs; ++i )
    {
        OptixBuildInput optixBuildInput = {};

        optixBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        optixBuildInput.triangleArray.vertexBuffers = &ommBuildInput[i].texCoordBuffer;
        OptixVertexFormat vertexFormat              = OPTIX_VERTEX_FORMAT_NONE;
        switch( ommBuildInput[i].texCoordFormat )
        {
            case cuOmmBaking::TexCoordFormat::UV32_FLOAT2:
                vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT2;
                break;
            default:
                return OPTIX_ERROR_INVALID_VALUE;  // error
        }
        optixBuildInput.triangleArray.vertexFormat        = vertexFormat;
        optixBuildInput.triangleArray.vertexStrideInBytes = ommBuildInput[i].texCoordStrideInBytes;
        optixBuildInput.triangleArray.numVertices         = ommBuildInput[i].numTexCoords;

        optixBuildInput.triangleArray.numIndexTriplets   = ommBuildInput[i].numIndexTriplets;
        optixBuildInput.triangleArray.indexStrideInBytes = ommBuildInput[i].indexTripletStrideInBytes;
        OptixIndicesFormat indexFormat                   = OPTIX_INDICES_FORMAT_NONE;
        switch( ommBuildInput[i].indexFormat )
        {
            break;
            case cuOmmBaking::IndexFormat::NONE:
                indexFormat = OPTIX_INDICES_FORMAT_NONE;
                break;
            case cuOmmBaking::IndexFormat::I16_UINT:
                indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
                break;
            case cuOmmBaking::IndexFormat::I32_UINT:
                indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                break;
            default:
                return OPTIX_ERROR_INVALID_VALUE;  // error
        }
        optixBuildInput.triangleArray.indexFormat = indexFormat;
        optixBuildInput.triangleArray.indexBuffer = ommBuildInput[i].indexBuffer;

        optixBuildInput.triangleArray.numSbtRecords               = ommBuildInput[i].numTextures;
        optixBuildInput.triangleArray.sbtIndexOffsetBuffer        = ommBuildInput[i].textureIndexBuffer;
        optixBuildInput.triangleArray.sbtIndexOffsetSizeInBytes   = getIndexFormatSizeInBytes( ommBuildInput[i].textureIndexFormat );
        optixBuildInput.triangleArray.sbtIndexOffsetStrideInBytes = ommBuildInput[i].textureIndexStrideInBytes;

        size_t idx = flags.size();
        for( unsigned int j = 0; j < optixBuildInput.triangleArray.numSbtRecords; ++j )
            flags.emplace_back( 0 );
        optixBuildInput.triangleArray.flags = &flags[idx];

        optixBuildInput.triangleArray.opacityMicromap = optixOmm.getBuildInput( i );

        bakeInputs.emplace_back( optixBuildInput );
    }

    if( numBuildInputs )
    {
        OptixAccelBuildOptions accelOptions = {};

        CuBuffer<char> d_temp;

        CUDA_SYNC_CHECK();

        {
            accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_DISABLE_OPACITY_MICROMAPS;
            accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes gasBufferSizes;
            OPTIX_CHECK( optixAccelComputeMemoryUsage( m_context, &accelOptions, bakeInputs.data(), (uint32_t)bakeInputs.size(), &gasBufferSizes ) );


            CUDA_CHECK( m_gas.alloc( gasBufferSizes.outputSizeInBytes ) );
            CUDA_CHECK( d_temp.alloc( gasBufferSizes.tempSizeInBytes ) );

            OPTIX_CHECK( optixAccelBuild( m_context, 0, &accelOptions, bakeInputs.data(), (uint32_t)bakeInputs.size(), d_temp.get(), d_temp.byteSize(), m_gas.get(), m_gas.byteSize(), &m_gasHandle, nullptr, 0 ) );
        }

        CUDA_SYNC_CHECK();

        {
            // build an IAS with 2 instances, both referencing the same gas. One with VM enabled and the other with VM disabled.
            // the visibility mask is used to enable/disable VMs at runtime.

            accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;

            float mtrx[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };

            OptixInstance instances[2] = {};
            memcpy( instances[0].transform, mtrx, sizeof( float ) * 12 );
            instances[0].flags             = OPTIX_INSTANCE_FLAG_NONE;
            instances[0].sbtOffset         = 0;
            instances[0].traversableHandle = m_gasHandle;
            instances[0].visibilityMask    = VISIBILITY_MASK_OMM_ENABLED;  // VM enabled mask
            instances[0].instanceId        = 0;

            instances[1]                = instances[0];
            instances[1].visibilityMask = VISIBILITY_MASK_OMM_DISABLED;
            instances[1].flags          = OPTIX_INSTANCE_FLAG_DISABLE_OPACITY_MICROMAPS;

            CuBuffer<OptixInstance> d_instances;
            CUDA_CHECK( d_instances.alloc( 2 ) );
            CUDA_CHECK( d_instances.upload( instances ) );

            OptixBuildInput optixBuildInput              = {};
            optixBuildInput.type                         = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            optixBuildInput.instanceArray.instances      = d_instances.get();
            optixBuildInput.instanceArray.instanceStride = 0;
            optixBuildInput.instanceArray.numInstances   = 2;

            OptixAccelBufferSizes iasBufferSizes;
            OPTIX_CHECK( optixAccelComputeMemoryUsage( m_context, &accelOptions, &optixBuildInput, 1, &iasBufferSizes ) );

            CUDA_CHECK( m_ias.alloc( iasBufferSizes.outputSizeInBytes ) );
            CUDA_CHECK( d_temp.allocIfRequired( iasBufferSizes.tempSizeInBytes ) );

            OPTIX_CHECK( optixAccelBuild( m_context, 0, &accelOptions, &optixBuildInput, 1, d_temp.get(), d_temp.byteSize(), m_ias.get(), m_ias.byteSize(), &m_iasHandle, nullptr, 0 ) );
        }

        CUDA_SYNC_CHECK();
    }

    return OPTIX_SUCCESS;
}

OptixResult OptixOmmScene::buildSBT( const cuOmmBaking::BakeInputDesc* ommBuildInput, unsigned int numBuildInputs )
{
    // set up SBT
    RayGenSbtRecord rgSBT;
    OPTIX_CHECK( optixSbtRecordPackHeader( m_rgProgramGroup, &rgSBT ) );
    MissSbtRecord msSBT[2];
    OPTIX_CHECK( optixSbtRecordPackHeader( m_msProgramGroup[0], &msSBT[0] ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( m_msProgramGroup[1], &msSBT[1] ) );
    HitSbtRecord hitSBT[2];
    OPTIX_CHECK( optixSbtRecordPackHeader( m_hitgroupProgramGroup[0], &hitSBT[0] ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( m_hitgroupProgramGroup[1], &hitSBT[1] ) );

    std::vector<DCSbtRecord>  dcSbtRecords;
    std::vector<HitSbtRecord> hitSbtRecords;
    for( unsigned int i = 0; i < numBuildInputs; ++i )
    {
        for( unsigned int j = 0; j < ommBuildInput[i].numTextures; ++j )
        {
            HitSbtData data = {};
            if( ommBuildInput[i].textures )
            {
                data.texture.tex                = ommBuildInput[i].textures[j].cuda.texObject;
                data.texture.transparencyCutoff = ommBuildInput[i].textures[j].cuda.transparencyCutoff;
                data.texture.opacityCutoff      = ommBuildInput[i].textures[j].cuda.opacityCutoff;
                data.texture.alphaMode          = ommBuildInput[i].textures[j].cuda.alphaMode;

                cudaChannelFormatDesc chanDesc = {};
                cudaResourceDesc      resDesc = {};

                CUDA_CHECK( cudaGetTextureObjectResourceDesc( &resDesc, data.texture.tex ) );

                switch( resDesc.resType )
                {
                case cudaResourceTypeArray: {
                    CUDA_CHECK( cudaGetChannelDesc( &chanDesc, resDesc.res.array.array ) );
                }
                break;
                case cudaResourceTypeMipmappedArray: {
                    cudaArray_t d_topLevelArray;
                    CUDA_CHECK( cudaGetMipmappedArrayLevel( &d_topLevelArray, resDesc.res.mipmap.mipmap, 0 ) );
                    CUDA_CHECK( cudaGetChannelDesc( &chanDesc, d_topLevelArray ) );
                }
                break;
                case cudaResourceTypePitch2D: {
                    chanDesc = resDesc.res.pitch2D.desc;
                }
                break;
                case cudaResourceTypeLinear:
                default:
                    break;
                };

                cudaTextureDesc texDesc;
                CUDA_CHECK( cudaGetTextureObjectTextureDesc( &texDesc, data.texture.tex ) );

                data.texture.readMode = texDesc.readMode;
                data.texture.chanDesc = chanDesc;
            }

            data.desc = ommBuildInput[i];

            HitSbtRecord hitRecord[2] = {};
            hitRecord[0]              = hitSBT[0];
            hitRecord[1]              = hitSBT[1];
            hitRecord[0].data         = data;
            hitRecord[1].data         = data;

            hitSbtRecords.emplace_back( hitRecord[0] );
            hitSbtRecords.emplace_back( hitRecord[1] );
        }
    }


    CUDA_CHECK( m_hitSbtRecords.allocAndUpload( hitSbtRecords ) );
    CUDA_CHECK( m_dcSbtRecords.allocAndUpload( dcSbtRecords ) );

    CUDA_CHECK( m_raygenSbtRecords.alloc( 1 ) );
    CUDA_CHECK( m_missSbtRecords.alloc( 2 ) );

    CUDA_CHECK( m_raygenSbtRecords.upload( &rgSBT ) );
    CUDA_CHECK( m_missSbtRecords.upload( msSBT ) );


    m_sbt.raygenRecord                 = m_raygenSbtRecords.get();
    m_sbt.missRecordBase               = m_missSbtRecords.get();
    m_sbt.missRecordStrideInBytes      = (unsigned int)sizeof( MissSbtRecord );
    m_sbt.missRecordCount              = (unsigned int)m_missSbtRecords.count();
    m_sbt.hitgroupRecordBase           = m_hitSbtRecords.get();
    m_sbt.hitgroupRecordStrideInBytes  = (unsigned int)sizeof( HitSbtRecord );
    m_sbt.hitgroupRecordCount          = (unsigned int)m_hitSbtRecords.count();
    m_sbt.callablesRecordBase          = m_dcSbtRecords.get();
    m_sbt.callablesRecordStrideInBytes = (unsigned int)sizeof( DCSbtRecord );
    m_sbt.callablesRecordCount         = (unsigned int)m_dcSbtRecords.count();

    return OPTIX_SUCCESS;
}

OptixResult OptixOmmScene::buildPipeline( const char* optixirInput, const size_t optixirInputSize )
{
    OptixTraversableGraphFlags traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;

    // Compile modules
    OptixModuleCompileOptions moduleCompileOptions{};
    otk::configModuleCompileOptions( moduleCompileOptions );
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    OptixPipelineCompileOptions pipelineCompileOptions      = {};
    pipelineCompileOptions.usesMotionBlur                   = false;
    pipelineCompileOptions.traversableGraphFlags            = traversableGraphFlags;
    pipelineCompileOptions.numPayloadValues                 = 5;
    pipelineCompileOptions.numAttributeValues               = 2;
    pipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;  // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags           = (unsigned int)OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipelineCompileOptions.allowOpacityMicromaps            = true;

    if( optixirInput )
    {
        OPTIX_CHECK( optixModuleCreate( m_context, &moduleCompileOptions, &pipelineCompileOptions, optixirInput, optixirInputSize, 0, 0, &m_moduleDC ) );
    }

    OPTIX_CHECK( optixModuleCreate( m_context, &moduleCompileOptions, &pipelineCompileOptions, OptiXKernelsCudaText(), OptiXKernelsCudaSize, 0, 0, &m_module));

    // Set up program groups

    std::vector<OptixProgramGroup> programs;

    OptixProgramGroupOptions programGroupOptions = {};

    OptixProgramGroupDesc rgProgramGroupDesc    = {};
    rgProgramGroupDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgProgramGroupDesc.raygen.module            = m_module;
    rgProgramGroupDesc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK( optixProgramGroupCreate( m_context, &rgProgramGroupDesc, 1, &programGroupOptions, 0, 0, &m_rgProgramGroup ) );

    OptixProgramGroupDesc msProgramGroupDesc  = {};
    msProgramGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msProgramGroupDesc.miss.module            = m_module;
    msProgramGroupDesc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK( optixProgramGroupCreate( m_context, &msProgramGroupDesc, 1, &programGroupOptions, 0, 0, &m_msProgramGroup[0] ) );
    msProgramGroupDesc.miss.entryFunctionName = "__miss__validate__ms";
    OPTIX_CHECK( optixProgramGroupCreate( m_context, &msProgramGroupDesc, 1, &programGroupOptions, 0, 0, &m_msProgramGroup[1] ) );

    OptixProgramGroupDesc hitgroupProgramGroupDesc        = {};
    hitgroupProgramGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupProgramGroupDesc.hitgroup.moduleCH            = m_module;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroupProgramGroupDesc.hitgroup.moduleAH            = m_module;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    OPTIX_CHECK( optixProgramGroupCreate( m_context, &hitgroupProgramGroupDesc, 1, &programGroupOptions, 0, 0, &m_hitgroupProgramGroup[0] ) );

    hitgroupProgramGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupProgramGroupDesc.hitgroup.moduleCH            = m_module;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__validate__ch";
    hitgroupProgramGroupDesc.hitgroup.moduleAH            = m_module;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameAH = "__anyhit__validate__ah";
    OPTIX_CHECK( optixProgramGroupCreate( m_context, &hitgroupProgramGroupDesc, 1, &programGroupOptions, 0, 0, &m_hitgroupProgramGroup[1] ) );

    programs.push_back( m_rgProgramGroup );
    programs.push_back( m_msProgramGroup[0] );
    programs.push_back( m_msProgramGroup[1] );
    programs.push_back( m_hitgroupProgramGroup[0] );
    programs.push_back( m_hitgroupProgramGroup[1] );

    // Link pipeline
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth            = 5;

    OPTIX_CHECK( optixPipelineCreate( m_context, &pipelineCompileOptions, &pipelineLinkOptions, programs.data(), (uint32_t)programs.size(), 0, 0, &m_pipeline ) );

    return OPTIX_SUCCESS;
}
