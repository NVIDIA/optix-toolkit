//
//  Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include <OptiXToolkit/DemandGeometry/Mocks/MockOptix.h>

#include <optix_function_table_definition.h>

namespace otk {
namespace testing {

MockOptix* g_mockOptix{};

// These definitions are placed here to speed up compilation of clients of the mock class.
// See <https://google.github.io/googletest/gmock_cook_book.html#making-the-compilation-faster>
MockOptix::MockOptix()
{
    initMockOptix( *this );
}
MockOptix::~MockOptix()
{
    g_mockOptix = nullptr;
}

void initMockOptix( MockOptix& mock )
{
    // mock OptiX already initialized?
    if( g_mockOptix != nullptr )
    {
        return;
    }

    // zero out any existing function table entries
    g_optixFunctionTable = OptixFunctionTable{};

    g_mockOptix = &mock;

    g_optixFunctionTable.optixGetErrorName = []( OptixResult result ) { return g_mockOptix->getErrorName( result ); };
    g_optixFunctionTable.optixGetErrorString = []( OptixResult result ) { return g_mockOptix->getErrorString( result ); };
    g_optixFunctionTable.optixDeviceContextCreate = []( CUcontext fromContext, const OptixDeviceContextOptions* options,
                                                        OptixDeviceContext* context ) {
        return g_mockOptix->deviceContextCreate( fromContext, options, context );
    };
    g_optixFunctionTable.optixDeviceContextDestroy = []( OptixDeviceContext context ) {
        return g_mockOptix->deviceContextDestroy( context );
    };
    g_optixFunctionTable.optixDeviceContextGetProperty = []( OptixDeviceContext context, OptixDeviceProperty property,
                                                             void* value, size_t sizeInBytes ) {
        return g_mockOptix->deviceContextGetProperty( context, property, value, sizeInBytes );
    };
    g_optixFunctionTable.optixDeviceContextSetLogCallback = []( OptixDeviceContext context, OptixLogCallback callbackFunction,
                                                                void* callbackData, unsigned int callbackLevel ) {
        return g_mockOptix->deviceContextSetLogCallback( context, callbackFunction, callbackData, callbackLevel );
    };
    g_optixFunctionTable.optixDeviceContextSetCacheEnabled = []( OptixDeviceContext context, int enabled ) {
        return g_mockOptix->deviceContextSetCacheEnabled( context, enabled );
    };
    g_optixFunctionTable.optixDeviceContextSetCacheLocation = []( OptixDeviceContext context, const char* location ) {
        return g_mockOptix->deviceContextSetCacheLocation( context, location );
    };
    g_optixFunctionTable.optixDeviceContextSetCacheDatabaseSizes = []( OptixDeviceContext context, size_t lowWaterMark,
                                                                       size_t highWaterMark ) {
        return g_mockOptix->deviceContextSetCacheDatabaseSizes( context, lowWaterMark, highWaterMark );
    };
    g_optixFunctionTable.optixDeviceContextGetCacheEnabled = []( OptixDeviceContext context, int* enabled ) {
        return g_mockOptix->deviceContextGetCacheEnabled( context, enabled );
    };
    g_optixFunctionTable.optixDeviceContextGetCacheLocation = []( OptixDeviceContext context, char* location, size_t locationSize ) {
        return g_mockOptix->deviceContextGetCacheLocation( context, location, locationSize );
    };
    g_optixFunctionTable.optixDeviceContextGetCacheDatabaseSizes = []( OptixDeviceContext context, size_t* lowWaterMark,
                                                                       size_t* highWaterMark ) {
        return g_mockOptix->deviceContextGetCacheDatabaseSizes( context, lowWaterMark, highWaterMark );
    };
#if OPTIX_VERSION >= 70700
    g_optixFunctionTable.optixModuleCreate = []( OptixDeviceContext context, const OptixModuleCompileOptions* moduleCompileOptions,
                                                 const OptixPipelineCompileOptions* pipelineCompileOptions, const char* input,
                                                 size_t inputSize, char* logString, size_t* logStringSize, OptixModule* module ) {
        return g_mockOptix->moduleCreate( context, moduleCompileOptions, pipelineCompileOptions, input, inputSize,
                                          logString, logStringSize, module );
    };
#else
    g_optixFunctionTable.optixModuleCreateFromPTX =
        []( OptixDeviceContext context, const OptixModuleCompileOptions* moduleCompileOptions,
            const OptixPipelineCompileOptions* pipelineCompileOptions, const char* PTX, size_t PTXsize, char* logString,
            size_t* logStringSize, OptixModule* module ) {
            return g_mockOptix->moduleCreateFromPTX( context, moduleCompileOptions, pipelineCompileOptions, PTX,
                                                     PTXsize, logString, logStringSize, module );
        };
#endif
#if OPTIX_VERSION >= 70400
#if OPTIX_VERSION >= 70700
    g_optixFunctionTable.optixModuleCreateWithTasks =
        []( OptixDeviceContext context, const OptixModuleCompileOptions* moduleCompileOptions,
            const OptixPipelineCompileOptions* pipelineCompileOptions, const char* input, size_t inputSize,
            char* logString, size_t* logStringSize, OptixModule* module, OptixTask* firstTask ) {
            return g_mockOptix->moduleCreateWithTasks( context, moduleCompileOptions, pipelineCompileOptions, input,
                                                       inputSize, logString, logStringSize, module, firstTask );
        };
#else
    g_optixFunctionTable.optixModuleCreateFromPTXWithTasks =
        []( OptixDeviceContext context, const OptixModuleCompileOptions* moduleCompileOptions,
            const OptixPipelineCompileOptions* pipelineCompileOptions, const char* PTX, size_t PTXsize, char* logString,
            size_t* logStringSize, OptixModule* module, OptixTask* firstTask ) {
            return g_mockOptix->moduleCreateFromPTXWithTasks( context, moduleCompileOptions, pipelineCompileOptions, PTX,
                                                              PTXsize, logString, logStringSize, module, firstTask );
        };
#endif
    g_optixFunctionTable.optixModuleGetCompilationState = []( OptixModule module, OptixModuleCompileState* state ) {
        return g_mockOptix->moduleGetCompilationState( module, state );
    };
#endif
    g_optixFunctionTable.optixModuleDestroy = []( OptixModule module ) { return g_mockOptix->moduleDestroy( module ); };
    g_optixFunctionTable.optixBuiltinISModuleGet = []( OptixDeviceContext context, const OptixModuleCompileOptions* moduleCompileOptions,
                                                       const OptixPipelineCompileOptions* pipelineCompileOptions,
                                                       const OptixBuiltinISOptions* builtinISOptions, OptixModule* builtinModule ) {
        return g_mockOptix->builtinISModuleGet( context, moduleCompileOptions, pipelineCompileOptions, builtinISOptions, builtinModule );
    };
#if OPTIX_VERSION >= 70400
    g_optixFunctionTable.optixTaskExecute = []( OptixTask task, OptixTask* additionalTasks, unsigned int maxNumAdditionalTasks,
                                                unsigned int* numAdditionalTasksCreated ) {
        return g_mockOptix->taskExecute( task, additionalTasks, maxNumAdditionalTasks, numAdditionalTasksCreated );
    };
#endif
    g_optixFunctionTable.optixProgramGroupCreate =
        []( OptixDeviceContext context, const OptixProgramGroupDesc* programDescriptions, unsigned int numProgramGroups,
            const OptixProgramGroupOptions* options, char* logString, size_t* logStringSize, OptixProgramGroup* programGroups ) {
            return g_mockOptix->programGroupCreate( context, programDescriptions, numProgramGroups, options, logString,
                                                    logStringSize, programGroups );
        };
    g_optixFunctionTable.optixProgramGroupDestroy = []( OptixProgramGroup group ) {
        return g_mockOptix->programGroupDestroy( group );
    };
#if OPTIX_VERSION >= 70700
    g_optixFunctionTable.optixProgramGroupGetStackSize = []( OptixProgramGroup programGroup,
                                                             OptixStackSizes* stackSizes, OptixPipeline pipeline ) {
        return g_mockOptix->programGroupGetStackSize( programGroup, stackSizes, pipeline );
    };
#else
    g_optixFunctionTable.optixProgramGroupGetStackSize = []( OptixProgramGroup programGroup, OptixStackSizes* stackSizes ) {
        return g_mockOptix->programGroupGetStackSize( programGroup, stackSizes );
    };
#endif
    g_optixFunctionTable.optixPipelineCreate =
        []( OptixDeviceContext context, const OptixPipelineCompileOptions* pipelineCompileOptions,
            const OptixPipelineLinkOptions* pipelineLinkOptions, const OptixProgramGroup* programGroups,
            unsigned int numProgramGroups, char* logString, size_t* logStringSize, OptixPipeline* pipeline ) {
            return g_mockOptix->pipelineCreate( context, pipelineCompileOptions, pipelineLinkOptions, programGroups,
                                                numProgramGroups, logString, logStringSize, pipeline );
        };
    g_optixFunctionTable.optixPipelineDestroy = []( OptixPipeline pipeline ) {
        return g_mockOptix->pipelineDestroy( pipeline );
    };
    g_optixFunctionTable.optixPipelineSetStackSize = []( OptixPipeline pipeline, unsigned int directCallableStackSizeFromTraversal,
                                                         unsigned int directCallableStackSizeFromState,
                                                         unsigned int continuationStackSize, unsigned int maxTraversableGraphDepth ) {
        return g_mockOptix->pipelineSetStackSize( pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
                                                  continuationStackSize, maxTraversableGraphDepth );
    };
    g_optixFunctionTable.optixAccelComputeMemoryUsage = []( OptixDeviceContext context, const OptixAccelBuildOptions* accelOptions,
                                                            const OptixBuildInput* buildInputs, unsigned int numBuildInputs,
                                                            OptixAccelBufferSizes* bufferSizes ) {
        return g_mockOptix->accelComputeMemoryUsage( context, accelOptions, buildInputs, numBuildInputs, bufferSizes );
    };
    g_optixFunctionTable.optixAccelBuild =
        []( OptixDeviceContext context, CUstream stream, const OptixAccelBuildOptions* accelOptions,
            const OptixBuildInput* buildInputs, unsigned int numBuildInputs, CUdeviceptr tempBuffer, size_t tempBufferSizeInBytes,
            CUdeviceptr outputBuffer, size_t outputBufferSizeInBytes, OptixTraversableHandle* outputHandle,
            const OptixAccelEmitDesc* emittedProperties, unsigned int numEmittedProperties ) {
            return g_mockOptix->accelBuild( context, stream, accelOptions, buildInputs, numBuildInputs, tempBuffer,
                                            tempBufferSizeInBytes, outputBuffer, outputBufferSizeInBytes, outputHandle,
                                            emittedProperties, numEmittedProperties );
        };
#if OPTIX_VERSION >= 70600
    g_optixFunctionTable.optixAccelGetRelocationInfo = []( OptixDeviceContext context, OptixTraversableHandle handle,
                                                           OptixRelocationInfo* info ) {
        return g_mockOptix->accelGetRelocationInfo( context, handle, info );
    };
    g_optixFunctionTable.optixCheckRelocationCompatibility = []( OptixDeviceContext         context,
                                                                 const OptixRelocationInfo* info, int* compatible ) {
        return g_mockOptix->checkRelocationCompatibility( context, info, compatible );
    };
    g_optixFunctionTable.optixAccelRelocate = []( OptixDeviceContext context, CUstream stream,
                                                  const OptixRelocationInfo* info, const OptixRelocateInput* relocateInputs,
                                                  size_t numRelocateInputs, CUdeviceptr targetAccel,
                                                  size_t targetAccelSizeInBytes, OptixTraversableHandle* targetHandle ) {
        return g_mockOptix->accelRelocate( context, stream, info, relocateInputs, numRelocateInputs, targetAccel,
                                           targetAccelSizeInBytes, targetHandle );
    };
#else
    g_optixFunctionTable.optixAccelGetRelocationInfo = []( OptixDeviceContext context, OptixTraversableHandle handle,
                                                           OptixAccelRelocationInfo* info ) {
        return g_mockOptix->accelGetRelocationInfo( context, handle, info );
    };
    g_optixFunctionTable.optixAccelCheckRelocationCompatibility =
        []( OptixDeviceContext context, const OptixAccelRelocationInfo* info, int* compatible ) {
            return g_mockOptix->accelCheckRelocationCompatibility( context, info, compatible );
        };
    g_optixFunctionTable.optixAccelRelocate = []( OptixDeviceContext context, CUstream stream,
                                                  const OptixAccelRelocationInfo* info, CUdeviceptr instanceTraversablehandles,
                                                  size_t numInstanceTraversableHandles, CUdeviceptr targetAccel,
                                                  size_t targetAccelSizeInBytes, OptixTraversableHandle* targetHandle ) {
        return g_mockOptix->accelRelocate( context, stream, info, instanceTraversablehandles, numInstanceTraversableHandles,
                                           targetAccel, targetAccelSizeInBytes, targetHandle );
    };
#endif
    g_optixFunctionTable.optixAccelCompact = []( OptixDeviceContext context, CUstream stream,
                                                 OptixTraversableHandle inputHandle, CUdeviceptr outputBuffer,
                                                 size_t outputBufferSizeInBytes, OptixTraversableHandle* outputHandle ) {
        return g_mockOptix->accelCompact( context, stream, inputHandle, outputBuffer, outputBufferSizeInBytes, outputHandle );
    };
#if OPTIX_VERSION >= 70700
    g_optixFunctionTable.optixAccelEmitProperty = []( OptixDeviceContext context, CUstream stream, OptixTraversableHandle handle,
                                                      const OptixAccelEmitDesc* emittedProperty ) {
        return g_mockOptix->accelEmitProperty( context, stream, handle, emittedProperty );
    };
#endif
    g_optixFunctionTable.optixConvertPointerToTraversableHandle = []( OptixDeviceContext onDevice, CUdeviceptr pointer,
                                                                      OptixTraversableType    traversableType,
                                                                      OptixTraversableHandle* traversableHandle ) {
        return g_mockOptix->convertPointerToTraversableHandle( onDevice, pointer, traversableType, traversableHandle );
    };
#if OPTIX_VERSION >= 70600
    g_optixFunctionTable.optixOpacityMicromapArrayComputeMemoryUsage =
        []( OptixDeviceContext context, const OptixOpacityMicromapArrayBuildInput* buildInput, OptixMicromapBufferSizes* bufferSizes ) {
            return g_mockOptix->opacityMicromapArrayComputeMemoryUsage( context, buildInput, bufferSizes );
        };
    g_optixFunctionTable.optixOpacityMicromapArrayBuild = []( OptixDeviceContext context, CUstream stream,
                                                              const OptixOpacityMicromapArrayBuildInput* buildInput,
                                                              const OptixMicromapBuffers*                buffers ) {
        return g_mockOptix->opacityMicromapArrayBuild( context, stream, buildInput, buffers );
    };
    g_optixFunctionTable.optixOpacityMicromapArrayGetRelocationInfo =
        []( OptixDeviceContext context, CUdeviceptr opacityMicromapArray, OptixRelocationInfo* info ) {
            return g_mockOptix->opacityMicromapArrayGetRelocationInfo( context, opacityMicromapArray, info );
        };
    g_optixFunctionTable.optixOpacityMicromapArrayRelocate =
        []( OptixDeviceContext context, CUstream stream, const OptixRelocationInfo* info,
            CUdeviceptr targetOpacityMicromapArray, size_t targetOpacityMicromapArraySizeInBytes ) {
            return g_mockOptix->opacityMicromapArrayRelocate( context, stream, info, targetOpacityMicromapArray,
                                                              targetOpacityMicromapArraySizeInBytes );
        };
#endif
#if OPTIX_VERSION >= 70700
    g_optixFunctionTable.optixDisplacementMicromapArrayComputeMemoryUsage =
        []( OptixDeviceContext context, const OptixDisplacementMicromapArrayBuildInput* buildInput,
            OptixMicromapBufferSizes* bufferSizes ) {
            return g_mockOptix->displacementMicromapArrayComputeMemoryUsage( context, buildInput, bufferSizes );
        };
    g_optixFunctionTable.optixDisplacementMicromapArrayBuild = []( OptixDeviceContext context, CUstream stream,
                                                                   const OptixDisplacementMicromapArrayBuildInput* buildInput,
                                                                   const OptixMicromapBuffers* buffers ) {
        return g_mockOptix->displacementMicromapArrayBuild( context, stream, buildInput, buffers );
    };
#endif
    g_optixFunctionTable.optixSbtRecordPackHeader = []( OptixProgramGroup programGroup, void* sbtRecordHeaderHostPointer ) {
        return g_mockOptix->sbtRecordPackHeader( programGroup, sbtRecordHeaderHostPointer );
    };
    g_optixFunctionTable.optixLaunch = []( OptixPipeline pipeline, CUstream stream, CUdeviceptr pipelineParams,
                                           size_t pipelineParamsSize, const OptixShaderBindingTable* sbt,
                                           unsigned int width, unsigned int height, unsigned int depth ) {
        return g_mockOptix->launch( pipeline, stream, pipelineParams, pipelineParamsSize, sbt, width, height, depth );
    };
    g_optixFunctionTable.optixDenoiserCreate = []( OptixDeviceContext context, OptixDenoiserModelKind modelKind,
                                                   const OptixDenoiserOptions* options, OptixDenoiser* returnHandle ) {
        return g_mockOptix->denoiserCreate( context, modelKind, options, returnHandle );
    };
    g_optixFunctionTable.optixDenoiserDestroy = []( OptixDenoiser handle ) {
        return g_mockOptix->denoiserDestroy( handle );
    };
    g_optixFunctionTable.optixDenoiserComputeMemoryResources = []( const OptixDenoiser handle, unsigned int maximumInputWidth,
                                                                   unsigned int maximumInputHeight, OptixDenoiserSizes* returnSizes ) {
        return g_mockOptix->denoiserComputeMemoryResources( handle, maximumInputWidth, maximumInputHeight, returnSizes );
    };
    g_optixFunctionTable.optixDenoiserSetup = []( OptixDenoiser denoiser, CUstream stream, unsigned int inputWidth,
                                                  unsigned int inputHeight, CUdeviceptr state, size_t stateSizeInBytes,
                                                  CUdeviceptr scratch, size_t scratchSizeInBytes ) {
        return g_mockOptix->denoiserSetup( denoiser, stream, inputWidth, inputHeight, state, stateSizeInBytes, scratch,
                                           scratchSizeInBytes );
    };
    g_optixFunctionTable.optixDenoiserInvoke = []( OptixDenoiser denoiser, CUstream stream, const OptixDenoiserParams* params,
                                                   CUdeviceptr denoiserState, size_t denoiserStateSizeInBytes,
                                                   const OptixDenoiserGuideLayer* guideLayer, const OptixDenoiserLayer* layers,
                                                   unsigned int numLayers, unsigned int inputOffsetX, unsigned int inputOffsetY,
                                                   CUdeviceptr scratch, size_t scratchSizeInBytes ) {
        return g_mockOptix->denoiserInvoke( denoiser, stream, params, denoiserState, denoiserStateSizeInBytes, guideLayer,
                                            layers, numLayers, inputOffsetX, inputOffsetY, scratch, scratchSizeInBytes );
    };
    g_optixFunctionTable.optixDenoiserComputeIntensity = []( OptixDenoiser handle, CUstream stream,
                                                             const OptixImage2D* inputImage, CUdeviceptr outputIntensity,
                                                             CUdeviceptr scratch, size_t scratchSizeInBytes ) {
        return g_mockOptix->denoiserComputeIntensity( handle, stream, inputImage, outputIntensity, scratch, scratchSizeInBytes );
    };
    g_optixFunctionTable.optixDenoiserComputeAverageColor = []( OptixDenoiser handle, CUstream stream,
                                                                const OptixImage2D* inputImage, CUdeviceptr outputAverageColor,
                                                                CUdeviceptr scratch, size_t scratchSizeInBytes ) {
        return g_mockOptix->denoiserComputeAverageColor( handle, stream, inputImage, outputAverageColor, scratch, scratchSizeInBytes );
    };
    g_optixFunctionTable.optixDenoiserCreateWithUserModel = []( OptixDeviceContext context, const void* data,
                                                                size_t dataSizeInBytes, OptixDenoiser* returnHandle ) {
        return g_mockOptix->denoiserCreateWithUserModel( context, data, dataSizeInBytes, returnHandle );
    };
}

}  // namespace testing
}  // namespace otk
