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

#include "testSia.h"

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#define _USE_MATH_DEFINES
#include <math.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <cuda_runtime.h>
#include "CuBuffer.h"

#include <testSiaKernelsCuda.h>

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif

#define CUDA_THROW( x )                                                                                                                                                                                                                                                                \
    {                                                                                                                                                                                                                                                                                  \
        cudaError_t err = (cudaError_t)( x );                                                                                                                                                                                                                                          \
        ASSERT_TRUE( err == cudaSuccess );                                                                                                                                                                                                                                             \
        if( err )                                                                                                                                                                                                                                                                      \
            throw std::runtime_error( "Cuda Runtime Api Error" );                                                                                                                                                                                                                      \
    };

#define OPTIX_THROW( x )                                                                                                                                                                                                                                                               \
    {                                                                                                                                                                                                                                                                                  \
        OptixResult err = (OptixResult)( x );                                                                                                                                                                                                                                          \
        ASSERT_TRUE( err == OPTIX_SUCCESS );                                                                                                                                                                                                                                           \
        if( err )                                                                                                                                                                                                                                                                      \
            throw std::runtime_error( "Optix Error" );                                                                                                                                                                                                                                 \
    };

cudaError_t launchCudaValidate( const Params params, unsigned int size, cudaStream_t stream );

namespace {  // anonymous

#if 0
float3 operator*( float a, float3 b )
{
    return { a * b.x, a * b.y, a * b.z };
}

float3 operator+( float3 a, float3 b )
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}
#endif

float dot( float4 a, float4 b )
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

Matrix3x4 operator*( Matrix3x4 a, Matrix3x4 b )
{
    Matrix3x4 c;

    float4 c0 = { b.row0.x, b.row1.x, b.row2.x, 0.f };
    float4 c1 = { b.row0.y, b.row1.y, b.row2.y, 0.f };
    float4 c2 = { b.row0.z, b.row1.z, b.row2.z, 0.f };
    float4 c3 = { b.row0.w, b.row1.w, b.row2.w, 1.f };

    c.row0 = { dot( a.row0, c0 ), dot( a.row0, c1 ), dot( a.row0, c2 ), dot( a.row0, c3 ) };
    c.row1 = { dot( a.row1, c0 ), dot( a.row1, c1 ), dot( a.row1, c2 ), dot( a.row1, c3 ) };
    c.row2 = { dot( a.row2, c0 ), dot( a.row2, c1 ), dot( a.row2, c2 ), dot( a.row2, c3 ) };

    return c;
}

float4 operator*( Matrix3x4 a, float4 b )
{
    return { dot( a.row0, b ), dot( a.row1, b ), dot( a.row2, b ), b.w };
}

const float time = 0.66f;

class SelfIntersectionAvoidanceTest : public testing::Test
{
protected:
    
    struct Transform
    {
        OptixTransformType type;
        union
        {
            OptixSRTMotionTransform    srt;
            OptixStaticTransform       mtrx;
            OptixMatrixMotionTransform mmtrx;
            OptixInstance              inst;
        };
    };

    static std::vector<float3> getTriangle()
    {
        // Centered around the origin to increase errors
        std::vector<float3> vertices;
        float3 v0 = { -1  , -1  , 0 };
        float3 v1 = { 0   , 0   , 1 };
        float3 v2 = { 0.5f, 0.5f, -0.5f };

        // Add some motion in either direction

        v0.y -= time;
        vertices.push_back( v0 );
        v0.y += (1.f - time);
        vertices.push_back( v0 );

        v1.y -= time;
        vertices.push_back( v1 );
        v1.y += ( 1.f - time );
        vertices.push_back( v1 );

        v2.y -= time;
        vertices.push_back( v2 );
        v2.y += ( 1.f - time );
        vertices.push_back( v2 );
        return vertices;
    }

    // turn srt into matrix
    static Matrix3x4 getMatrix( OptixSRTData srt )
    {
        Matrix3x4 s = {}, r = {}, t = {};
        s.row0 = { srt.sx,  srt.a,  srt.b, srt.pvx };
        s.row1 = {      0, srt.sy,  srt.c, srt.pvy };
        s.row2 = {      0,      0, srt.sz, srt.pvz };

        float sqw = srt.qw * srt.qw;
        float sqx = srt.qx * srt.qx;
        float sqy = srt.qy * srt.qy;
        float sqz = srt.qz * srt.qz;

        float invs = 1.f / ( sqx + sqy + sqz + sqw );
        r.row0.x = ( sqx - sqy - sqz + sqw ) * invs;
        r.row1.y = ( -sqx + sqy - sqz + sqw ) * invs;
        r.row2.z = ( -sqx - sqy + sqz + sqw ) * invs;

        float tmp1 = srt.qx * srt.qy;
        float tmp2 = srt.qz * srt.qw;
        r.row1.x = 2.0f * ( tmp1 + tmp2 ) * invs;
        r.row0.y = 2.0f * ( tmp1 - tmp2 ) * invs;

        tmp1 = srt.qx * srt.qz;
        tmp2 = srt.qy * srt.qw;
        r.row2.x = 2.0f * ( tmp1 - tmp2 ) * invs;
        r.row0.z = 2.0f * ( tmp1 + tmp2 ) * invs;
        tmp1 = srt.qy * srt.qz;
        tmp2 = srt.qx * srt.qw;
        r.row2.y = 2.0f * ( tmp1 + tmp2 ) * invs;
        r.row1.z = 2.0f * ( tmp1 - tmp2 ) * invs;

        t.row0 = { 1.f, 0.f, 0.f, srt.tx };
        t.row1 = { 0.f, 1.f, 0.f, srt.ty };
        t.row2 = { 0.f, 0.f, 1.f, srt.tz };

        return t * r * s;
    }

    // get some arbitrary test transform
    static OptixSRTData getSrtData()
    {
        OptixSRTData data = {};

        data.sx = 2.f; data.a = 3.f;   data.b = 5.f; data.pvx =  7.f;
                      data.sy = 11.f; data.c = 13.f; data.pvy = 17.f;
                                     data.sz = 23.f; data.pvz = 29.f;

        float phi   = 43.f * M_PI;
        float theta = 47.f * M_PI;
        float eta   = 51.f * M_PI;

        const float sp = sinf( phi );
        const float cp = cosf( phi );

        const float se = sinf( eta / 2.f );
        const float ce = cosf( eta / 2.f );

        data.qx = se * cp * cosf( theta );
        data.qy = se * cp * sinf( theta );
        data.qz = se * sp;
        data.qw = ce;

        data.tx = 31.f;
        data.ty = 37.f;
        data.tz = 41.f;

        return data;
    }

    static Transform getSrt()
    {
        OptixSRTData data = getSrtData();

        // recenter to increase relative errors
        float4 t = getMatrix( data ) * make_float4( 0, 0, 0, 1 );
        data.tx -= t.x;
        data.ty -= t.y;
        data.tz -= t.z;

        Transform transform = {};
        transform.type = OPTIX_TRANSFORM_TYPE_SRT_MOTION_TRANSFORM;
        transform.srt.motionOptions.numKeys = 2;
        transform.srt.motionOptions.timeBegin = 0.f;
        transform.srt.motionOptions.timeEnd = 1.f;
        transform.srt.srtData[0] = data;
        transform.srt.srtData[1] = data;

        // Add some motion in either direction
        transform.srt.srtData[0].tx -= (     time ) * 1.f;
        transform.srt.srtData[1].tx += ( 1 - time ) * 1.f;
        return transform;
    }

    static Transform getMotionMatrix()
    {
        Matrix3x4 mtrx = getMatrix( getSrtData() );

        // recenter to increase relative errors
        float4 t = mtrx * make_float4(0,0,0,1);
        mtrx.row0.w -= t.x;
        mtrx.row1.w -= t.y;
        mtrx.row2.w -= t.z;

        Transform transform = {};
        transform.type = OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM;
        transform.mmtrx.motionOptions.numKeys = 2;
        transform.mmtrx.motionOptions.timeBegin = 0.f;
        transform.mmtrx.motionOptions.timeEnd = 1.f;
        memcpy( transform.mmtrx.transform[0], &mtrx.row0.x, sizeof( float ) * 12 );
        memcpy( transform.mmtrx.transform[1], transform.mmtrx.transform[0], sizeof( float ) * 12 );

        // Add some motion in either direction
        transform.mmtrx.transform[0][3] -= (     time ) * 1.f;
        transform.mmtrx.transform[1][3] += ( 1 - time ) * 1.f;
        return transform;
    }

    static Transform getStaticMatrix()
    {
        Matrix3x4 mtrx = getMatrix( getSrtData() );

        // recenter to increase relative errors
        float4 t = mtrx * make_float4( 0, 0, 0, 1 );
        mtrx.row0.w -= t.x;
        mtrx.row1.w -= t.y;
        mtrx.row2.w -= t.z;

        Transform transform = {};
        transform.type = OPTIX_TRANSFORM_TYPE_STATIC_TRANSFORM;
        memcpy( transform.mtrx.transform, &mtrx.row0.x, sizeof( float ) * 12 );
        return transform;
    }

    static Transform getInstance()
    {
        Matrix3x4 mtrx = getMatrix( getSrtData() );

        // recenter to increase relative errors
        float4 t = mtrx * make_float4( 0, 0, 0, 1 );
        mtrx.row0.w -= t.x;
        mtrx.row1.w -= t.y;
        mtrx.row2.w -= t.z;

        Transform transform = {};
        transform.type = OPTIX_TRANSFORM_TYPE_INSTANCE;
        memcpy( transform.inst.transform, &mtrx.row0.x, sizeof( float ) * 12 );
        transform.inst.visibilityMask = 1u;
        return transform;
    }

    struct TestOptions
    {
        std::vector<Transform> transforms;        
    };

    void runTest( const TestOptions& opt, const std::string& /*imageNamePrefix*/ )
    {
        std::vector<float3> vertices = getTriangle();

        Params p = {};
        p.time = time;

        // Initialize CUDA runtime
        CUDA_THROW( cudaFree( 0 ) );

        // Create optix context
        OptixDeviceContextOptions optixOptions = {};

        std::vector<CUdeviceptr> pointers;

        OPTIX_THROW( optixInit() );

        CUcontext          cuCtx = nullptr;  // zero means take the current context
        OptixDeviceContext optixContext = {};
        OPTIX_THROW( optixDeviceContextCreate( cuCtx, &optixOptions, &optixContext ) );

        // build GAS

        CuBuffer<float3> d_vertices;
        d_vertices.allocAndUpload( vertices );

        CUdeviceptr d_vertexBuffer[2] = { d_vertices.get(), d_vertices.get() + sizeof( float3 ) };

        unsigned int flags = 0;
        OptixBuildInput optixBuildInput = {};
        optixBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        optixBuildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
        optixBuildInput.triangleArray.vertexBuffers       = d_vertexBuffer;
        optixBuildInput.triangleArray.numVertices         = 3;
        optixBuildInput.triangleArray.vertexStrideInBytes = sizeof( float3 ) * 2;
        optixBuildInput.triangleArray.numSbtRecords       = 1u;
        optixBuildInput.triangleArray.flags               = &flags;

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        accelOptions.motionOptions.numKeys = 2;
        accelOptions.motionOptions.timeBegin = 0.f;
        accelOptions.motionOptions.timeEnd   = 1.f;

        CuBuffer<char> d_temp;
        CuBuffer<char> m_gas;

        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_THROW( optixAccelComputeMemoryUsage( optixContext, &accelOptions, &optixBuildInput, 1u, &gasBufferSizes ) );

        CUDA_THROW( m_gas.alloc( gasBufferSizes.outputSizeInBytes ) );
        CUDA_THROW( d_temp.alloc( gasBufferSizes.tempSizeInBytes ) );

        OptixTraversableHandle gasHandle = {};
        OPTIX_THROW( optixAccelBuild( optixContext, 0, &accelOptions, &optixBuildInput, 1u, d_temp.get(), d_temp.byteSize(), m_gas.get(), m_gas.byteSize(), &gasHandle, nullptr, 0 ) );

        // build transform list
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;

        OptixTraversableHandle handle = gasHandle;
        for( const Transform& trns : opt.transforms )
        {
            TransformPtr ptr;
            ptr.type = trns.type;

            switch( trns.type )
            {
                case OPTIX_TRANSFORM_TYPE_INSTANCE:
                {
                    OptixInstance inst = trns.inst;
                    inst.traversableHandle = handle;

                    CuBuffer<OptixInstance> d_instances;
                    d_instances.allocAndUpload( 1u, &inst );

                    OptixBuildInput buildInput = {};
                    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
                    buildInput.instanceArray.instances = d_instances.get();
                    buildInput.instanceArray.numInstances = 1;
#if OPTIX_VERSION >= 70600
                    buildInput.instanceArray.instanceStride = 0;
#endif                    

                    OptixAccelBufferSizes iasBufferSizes;
                    OPTIX_THROW( optixAccelComputeMemoryUsage( optixContext, &accelOptions, &buildInput, 1, &iasBufferSizes ) );

                    CuBuffer<char> d_ias;

                    CUDA_THROW( d_ias.alloc( iasBufferSizes.outputSizeInBytes ) );
                    CUDA_THROW( d_temp.allocIfRequired( iasBufferSizes.tempSizeInBytes ) );

                    OptixTraversableHandle iasHandle = {};
                    OPTIX_THROW( optixAccelBuild( optixContext, 0, &accelOptions, &buildInput, 1, d_temp.get(), d_temp.byteSize(), d_ias.get(), d_ias.byteSize(), &iasHandle, nullptr, 0 ) );

                    ptr.inst = ( OptixInstance* )d_instances.get();
                    pointers.push_back( d_instances.release() );
                    pointers.push_back( d_ias.release() );

                    handle = iasHandle;
                } break;
                case OPTIX_TRANSFORM_TYPE_STATIC_TRANSFORM:
                {
                    OptixStaticTransform transform = trns.mtrx;
                    transform.child = handle;

                    Matrix3x4 m, im;
                    memcpy( &m, transform.transform, sizeof( float ) * 12 );
                    im = computeInverseAffine3x4( m );
                    memcpy( transform.invTransform, &im, sizeof( float ) * 12 );

                    CuBuffer<OptixStaticTransform> d_transform;
                    CUDA_THROW( d_transform.allocAndUpload( 1u, &transform ) );

                    OPTIX_THROW( optixConvertPointerToTraversableHandle( optixContext, d_transform.get(), OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM, &handle ) );

                    ptr.mtrx = ( OptixStaticTransform* )d_transform.get();
                    pointers.push_back( d_transform.release() );
                } break;
                case OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM:
                {
                    OptixMatrixMotionTransform transform = trns.mmtrx;
                    transform.child = handle;

                    CuBuffer<OptixMatrixMotionTransform> d_transform;
                    CUDA_THROW( d_transform.allocAndUpload( 1u, &transform ) );

                    OPTIX_THROW( optixConvertPointerToTraversableHandle( optixContext, d_transform.get(), OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM, &handle ) );

                    ptr.mmtrx = ( OptixMatrixMotionTransform* )d_transform.get();
                    pointers.push_back( d_transform.release() );
                } break;
                case OPTIX_TRANSFORM_TYPE_SRT_MOTION_TRANSFORM:
                {
                    OptixSRTMotionTransform transform = trns.srt;
                    transform.child = handle;

                    CuBuffer<OptixSRTMotionTransform> d_transform;
                    CUDA_THROW( d_transform.allocAndUpload( 1u, &transform ) );

                    OPTIX_THROW( optixConvertPointerToTraversableHandle( optixContext, d_transform.get(), OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM, &handle ) );

                    ptr.srt = ( OptixSRTMotionTransform* )d_transform.get();
                    pointers.push_back( d_transform.release() );
                }
                break;
                case OPTIX_TRANSFORM_TYPE_NONE: {
                    ASSERT_TRUE( false && "Unexpected transform" );
                    break;
                }
            }

            p.transforms[opt.transforms.size() - 1 - p.depth] = ptr;
            p.handles[p.depth++] = handle;
        }

        p.root = handle;

        OptixModuleCompileOptions moduleCompileOptions = {};
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        pipelineCompileOptions.usesMotionBlur                   = true;
        pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipelineCompileOptions.numPayloadValues                 = 1;
        pipelineCompileOptions.numAttributeValues               = 2;
        pipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
        pipelineCompileOptions.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        OptixModule ptxModule;
        OPTIX_THROW( optixModuleCreate( optixContext, &moduleCompileOptions, &pipelineCompileOptions, testSiaOptixCudaText(), testSiaOptixCudaSize, 0, 0, &ptxModule ) );

        OptixProgramGroupOptions programGroupOptions = {};

        OptixProgramGroupDesc rgProgramGroupDesc = {};
        rgProgramGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        rgProgramGroupDesc.raygen.module = ptxModule;
        rgProgramGroupDesc.raygen.entryFunctionName = "__raygen__rg";
        OptixProgramGroup rgProgramGroup;
        OPTIX_THROW( optixProgramGroupCreate( optixContext, &rgProgramGroupDesc, 1, &programGroupOptions, 0, 0, &rgProgramGroup ) );

        OptixProgramGroupDesc msProgramGroupDesc = {};
        msProgramGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        msProgramGroupDesc.miss.module = ptxModule;
        msProgramGroupDesc.miss.entryFunctionName = "__miss__ms";
        OptixProgramGroup msProgramGroup;
        OPTIX_THROW( optixProgramGroupCreate( optixContext, &msProgramGroupDesc, 1, &programGroupOptions, 0, 0, &msProgramGroup ) );

        OptixProgramGroupDesc hitgroupProgramGroupDesc = {};
        hitgroupProgramGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroupProgramGroupDesc.hitgroup.moduleCH = ptxModule;
        hitgroupProgramGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        OptixProgramGroup hitgroupProgramGroup;
        OPTIX_THROW( optixProgramGroupCreate( optixContext, &hitgroupProgramGroupDesc, 1, &programGroupOptions, 0, 0, &hitgroupProgramGroup ) );

        OptixProgramGroup        programGroups[] = { rgProgramGroup, msProgramGroup, hitgroupProgramGroup };

        OptixPipeline            optixPipeline;
        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = 2;
        OPTIX_THROW( optixPipelineCreate( optixContext, &pipelineCompileOptions, &pipelineLinkOptions, programGroups,
            sizeof( programGroups ) / sizeof( programGroups[0] ), 0, 0, &optixPipeline ) );

        // Calculate the stack sizes, so we can specify all parameters to optixPipelineSetStackSize.
        OptixStackSizes stack_sizes = {};
#if OPTIX_VERSION < 70700
        OPTIX_THROW( optixUtilAccumulateStackSizes( hitgroupProgramGroup, &stack_sizes ) );
        OPTIX_THROW( optixUtilAccumulateStackSizes( msProgramGroup, &stack_sizes ) );
        OPTIX_THROW( optixUtilAccumulateStackSizes( rgProgramGroup, &stack_sizes ) );
#else
        OPTIX_THROW( optixUtilAccumulateStackSizes( hitgroupProgramGroup, &stack_sizes, optixPipeline ) );
        OPTIX_THROW( optixUtilAccumulateStackSizes( msProgramGroup, &stack_sizes, optixPipeline ) );
        OPTIX_THROW( optixUtilAccumulateStackSizes( rgProgramGroup, &stack_sizes, optixPipeline ) );
#endif
        
        // We need to specify the max traversal depth.  
        uint32_t max_cc_depth = 0;
        uint32_t max_dc_depth = 0;
        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_THROW( optixUtilComputeStackSizes( &stack_sizes, 1, max_cc_depth, max_dc_depth, &direct_callable_stack_size_from_traversal,
            &direct_callable_stack_size_from_state, &continuation_stack_size ) );

        uint32_t max_traversable_graph_depth = p.depth + 1;
        OPTIX_THROW( optixPipelineSetStackSize( optixPipeline, direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state,
            continuation_stack_size, max_traversable_graph_depth ) );

        CuBuffer<RayGenSbtRecord>   d_rayGenData;
        CuBuffer<MissSbtRecord>     d_missData;
        CuBuffer<InstanceSbtRecord> d_hitData;

        RayGenSbtRecord rgSBT;
        OPTIX_THROW( optixSbtRecordPackHeader( rgProgramGroup, &rgSBT ) );
        MissSbtRecord msSBT;
        OPTIX_THROW( optixSbtRecordPackHeader( msProgramGroup, &msSBT ) );
        InstanceSbtRecord inSBT;
        OPTIX_THROW( optixSbtRecordPackHeader( hitgroupProgramGroup, &inSBT ) );

        d_rayGenData.allocAndUpload( 1, &rgSBT );
        d_missData.allocAndUpload( 1, &msSBT );
        d_hitData.allocAndUpload( 1, &inSBT );

        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord = d_rayGenData.get();
        sbt.missRecordBase = d_missData.get();
        sbt.missRecordStrideInBytes = ( unsigned int )sizeof( MissSbtRecord );
        sbt.missRecordCount = d_missData.count();
        sbt.hitgroupRecordBase = d_hitData.get();
        sbt.hitgroupRecordStrideInBytes = ( unsigned int )sizeof( InstanceSbtRecord );
        sbt.hitgroupRecordCount = d_hitData.count();

        uint32_t width  = 512;
        uint32_t height = 512;

        Stats stats = {};
        CuBuffer<Stats> d_stats;
        CUDA_THROW( d_stats.allocAndUpload( 1, &stats ) );

        CuBuffer<float2> d_bary;
        CUDA_THROW( d_bary.alloc( width * height ) );

        CuBuffer<SpawnPoint> d_spawn;
        CUDA_THROW( d_spawn.alloc( width* height ) );

        p.size = width * height;
        p.barys = (float2*)d_bary.get();
        p.stats = (Stats*)d_stats.get();
        p.vertices = ( float3* )d_vertices.get();
        p.spawn = (SpawnPoint*)d_spawn.get();

        CuBuffer<Params>   d_param;
        CUDA_THROW( d_param.allocAndUpload( 1, &p ) );

        // run optix offsetting test
        OPTIX_THROW( optixLaunch( optixPipeline, 0, d_param.get(), sizeof(Params), &sbt, width, height, /*depth=*/1));

        // validate against cuda offsetting
        CUDA_THROW( launchCudaValidate( p, width* height, 0 ) );

        // download the stats
        CUDA_THROW( d_stats.download( &m_stats ) );

        OPTIX_THROW( optixPipelineDestroy( optixPipeline ) );
        OPTIX_THROW( optixProgramGroupDestroy( rgProgramGroup ) );
        OPTIX_THROW( optixProgramGroupDestroy( msProgramGroup ) );
        OPTIX_THROW( optixProgramGroupDestroy( hitgroupProgramGroup ) );
        OPTIX_THROW( optixModuleDestroy( ptxModule ) );

        for( auto ptr : pointers )
            CUDA_THROW( cudaFree( (void*)ptr ) );

        OPTIX_THROW( optixDeviceContextDestroy( optixContext ) );

        double samples = width * height;
        double total = static_cast<double>( m_stats.frontMissBackMiss + m_stats.frontHitBackMiss + m_stats.frontMissBackHit + m_stats.frontHitBackHit );

        double prcTest = total / samples;
        double prcMM = m_stats.frontMissBackMiss / total;
        double prcHM = m_stats.frontHitBackMiss / total;
        double prcMH = m_stats.frontMissBackHit / total;
        double prcHH = m_stats.frontHitBackHit / total;

        EXPECT_GE( prcTest, 0.99 ); // majority of primary rays hit the test triangle
        EXPECT_EQ( prcHH, 0 );      // no self intersections
        EXPECT_EQ( prcHM, 0 );      // no flipped intersections
        EXPECT_LE( prcMM, 0.01 );   // few double-misses due to near triangle edge origins and grazing angle directions
        EXPECT_GE( prcMH, 0.99 );   // majority miss front but hit back as expected

        EXPECT_EQ( m_stats.contextContextFreeMissmatch, 0 );
        EXPECT_EQ( m_stats.optixCudaMissmatch, 0 );

        return;
    }

    Stats m_stats = {};
};

} // namespace

TEST_F( SelfIntersectionAvoidanceTest, Base )
{
    TestOptions opt = {};
    runTest( opt, "SelfIntersectionAvoidanceTest_Base" );
}

TEST_F( SelfIntersectionAvoidanceTest, Inst )
{
    TestOptions opt = {};
    opt.transforms.push_back( getInstance() );
    runTest( opt, "SelfIntersectionAvoidanceTest_Inst" );
}

TEST_F( SelfIntersectionAvoidanceTest, Mtrx )
{
    TestOptions opt = {};
    opt.transforms.push_back( getStaticMatrix() );
    runTest( opt, "SelfIntersectionAvoidanceTest_Mtrx" );
}

TEST_F( SelfIntersectionAvoidanceTest, Srt )
{
    TestOptions opt = {};
    opt.transforms.push_back( getSrt() );
    runTest( opt, "SelfIntersectionAvoidanceTest_Srt" );
}

TEST_F( SelfIntersectionAvoidanceTest, MMtrx )
{
    TestOptions opt = {};
    opt.transforms.push_back( getMotionMatrix() );
    runTest( opt, "SelfIntersectionAvoidanceTest_MMtrx" );
}

TEST_F( SelfIntersectionAvoidanceTest, MtrxInst )
{
    TestOptions opt = {};
    opt.transforms.push_back( getStaticMatrix() );
    opt.transforms.push_back( getInstance() );
    runTest( opt, "SelfIntersectionAvoidanceTest_MtrxInst" );
}

TEST_F( SelfIntersectionAvoidanceTest, SrtInst )
{
    TestOptions opt = {};
    opt.transforms.push_back( getSrt() );
    opt.transforms.push_back( getInstance() );
    runTest( opt, "SelfIntersectionAvoidanceTest_SrtInst" );
}

TEST_F( SelfIntersectionAvoidanceTest, MMtrxInst )
{
    TestOptions opt = {};
    opt.transforms.push_back( getMotionMatrix() );
    opt.transforms.push_back( getInstance() );
    runTest( opt, "SelfIntersectionAvoidanceTest_MMtrxInst" );
}

TEST_F( SelfIntersectionAvoidanceTest, DISABLED_InstMtrx )
{
    TestOptions opt = {};
    opt.transforms.push_back( getInstance() );
    opt.transforms.push_back( getStaticMatrix() );
    runTest( opt, "SelfIntersectionAvoidanceTest_InstMtrx" );
}

TEST_F( SelfIntersectionAvoidanceTest, DISABLED_InstSrt )
{
    TestOptions opt = {};
    opt.transforms.push_back( getInstance() );
    opt.transforms.push_back( getSrt() );
    runTest( opt, "SelfIntersectionAvoidanceTest_InstSrt" );
}

TEST_F( SelfIntersectionAvoidanceTest, DISABLED_InstMMtrx )
{
    TestOptions opt = {};
    opt.transforms.push_back( getInstance() );
    opt.transforms.push_back( getMotionMatrix() );
    runTest( opt, "SelfIntersectionAvoidanceTest_InstMMtrx" );
}

TEST_F( SelfIntersectionAvoidanceTest, InstInst )
{
    TestOptions opt = {};
    opt.transforms.push_back( getInstance() );
    opt.transforms.push_back( getInstance() );
    runTest( opt, "SelfIntersectionAvoidanceTest_InstInst" );
}
