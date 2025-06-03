// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "hello.h"

#include "otkHelloKernelCuda.h"

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Gui/Window.h>
#include <OptiXToolkit/OptiXMemory/CompileOptions.h>
#include <OptiXToolkit/Util/Logger.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <iostream>
#include <string>

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


void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit( 1 );
}


int main( int argc, char* argv[] )
{
    std::string outfile;
    int         width  = 512;
    int         height = 384;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i < argc - 1 )
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            otk::parseDimensions( dims_arg.c_str(), width, height );
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            OTK_ERROR_CHECK( cudaFree( 0 ) );

            CUcontext cuCtx = 0;  // zero means take the current context
            OTK_ERROR_CHECK( optixInit() );
            OptixDeviceContextOptions options = {};
            otk::util::setLogger(options);
            OTK_ERROR_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
        }

        //
        // Create module
        //
        OptixModule                 module                   = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options{};
            otk::configModuleCompileOptions( module_compile_options );
            pipeline_compile_options.usesMotionBlur        = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
            pipeline_compile_options.numPayloadValues      = 2;
            pipeline_compile_options.numAttributeValues    = 2;
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

            OTK_ERROR_CHECK_LOG( optixModuleCreate( context, &module_compile_options, &pipeline_compile_options,
                                                 draw_solid_colorCudaText(), draw_solid_colorCudaSize, LOG, &LOG_SIZE, &module ) );
        }

        //
        // Create program groups, including NULL miss and hitgroups
        //
        OptixProgramGroup raygen_prog_group = nullptr;
        OptixProgramGroup miss_prog_group   = nullptr;
        {
            OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc    = {};  //
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__draw_solid_color";
            OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, &raygen_prog_group_desc,
                                                       1,  // num program groups
                                                       &program_group_options, LOG, &LOG_SIZE, &raygen_prog_group ) );

            // Leave miss group's module and entryfunc name null
            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_MISS;
            OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( context, &miss_prog_group_desc,
                                                       1,  // num program groups
                                                       &program_group_options, LOG, &LOG_SIZE, &miss_prog_group ) );
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
            OTK_ERROR_CHECK_LOG( optixPipelineCreate( context, &pipeline_compile_options, &pipeline_link_options,
                                                   program_groups, sizeof( program_groups ) / sizeof( program_groups[0] ),
                                                   LOG, &LOG_SIZE, &pipeline ) );

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
            OTK_ERROR_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
            RayGenSbtRecord rg_sbt;
            OTK_ERROR_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
            rg_sbt.data = {0.462f, 0.725f, 0.f};
            OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( raygen_record ), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice ) );

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof( MissSbtRecord );
            OTK_ERROR_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
            MissSbtRecord ms_sbt;
            OTK_ERROR_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
            OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( miss_record ), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice ) );

            sbt.raygenRecord            = raygen_record;
            sbt.missRecordBase          = miss_record;
            sbt.missRecordStrideInBytes = sizeof( MissSbtRecord );
            sbt.missRecordCount         = 1;
        }

        otk::CUDAOutputBuffer<uchar4> output_buffer( otk::CUDAOutputBufferType::CUDA_DEVICE, width, height );

        //
        // launch
        //
        {
            CUstream stream;
            OTK_ERROR_CHECK( cudaStreamCreate( &stream ) );

            Params params;
            params.image       = output_buffer.map();
            params.image_width = width;

            CUdeviceptr d_param;
            OTK_ERROR_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
            OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_param ), &params, sizeof( params ), cudaMemcpyHostToDevice ) );

            OTK_ERROR_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, width, height, /*depth=*/1 ) );
            OTK_CUDA_SYNC_CHECK();

            output_buffer.unmap();
            OTK_ERROR_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );
        }

        //
        // Display results
        //
        {
            otk::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = width;
            buffer.height       = height;
            buffer.pixel_format = otk::BufferImageFormat::UNSIGNED_BYTE4;
            if( outfile.empty() )
                otk::displayBufferWindow( argv[0], buffer );
            else
                otk::saveImage( outfile.c_str(), buffer, false );
        }

        //
        // Cleanup
        //
        {
            OTK_ERROR_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord ) ) );
            OTK_ERROR_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase ) ) );

            OTK_ERROR_CHECK( optixPipelineDestroy( pipeline ) );
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
