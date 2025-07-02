// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
    OptixFunctionTable& funcTable = OPTIX_FUNCTION_TABLE_SYMBOL;
    funcTable                     = OptixFunctionTable{};

    g_mockOptix = &mock;

    funcTable.optixGetErrorName        = []( OptixResult result ) { return g_mockOptix->getErrorName( result ); };
    funcTable.optixGetErrorString      = []( OptixResult result ) { return g_mockOptix->getErrorString( result ); };
    funcTable.optixDeviceContextCreate = []( CUcontext fromContext, const OptixDeviceContextOptions* options,
                                             OptixDeviceContext* context ) {
        return g_mockOptix->deviceContextCreate( fromContext, options, context );
    };
    funcTable.optixDeviceContextDestroy = []( OptixDeviceContext context ) {
        return g_mockOptix->deviceContextDestroy( context );
    };
    funcTable.optixDeviceContextGetProperty = []( OptixDeviceContext context, OptixDeviceProperty property, void* value,
                                                  size_t sizeInBytes ) {
        return g_mockOptix->deviceContextGetProperty( context, property, value, sizeInBytes );
    };
    funcTable.optixDeviceContextSetLogCallback = []( OptixDeviceContext context, OptixLogCallback callbackFunction,
                                                     void* callbackData, unsigned int callbackLevel ) {
        return g_mockOptix->deviceContextSetLogCallback( context, callbackFunction, callbackData, callbackLevel );
    };
    funcTable.optixDeviceContextSetCacheEnabled = []( OptixDeviceContext context, int enabled ) {
        return g_mockOptix->deviceContextSetCacheEnabled( context, enabled );
    };
    funcTable.optixDeviceContextSetCacheLocation = []( OptixDeviceContext context, const char* location ) {
        return g_mockOptix->deviceContextSetCacheLocation( context, location );
    };
    funcTable.optixDeviceContextSetCacheDatabaseSizes = []( OptixDeviceContext context, size_t lowWaterMark, size_t highWaterMark ) {
        return g_mockOptix->deviceContextSetCacheDatabaseSizes( context, lowWaterMark, highWaterMark );
    };
    funcTable.optixDeviceContextGetCacheEnabled = []( OptixDeviceContext context, int* enabled ) {
        return g_mockOptix->deviceContextGetCacheEnabled( context, enabled );
    };
    funcTable.optixDeviceContextGetCacheLocation = []( OptixDeviceContext context, char* location, size_t locationSize ) {
        return g_mockOptix->deviceContextGetCacheLocation( context, location, locationSize );
    };
    funcTable.optixDeviceContextGetCacheDatabaseSizes = []( OptixDeviceContext context, size_t* lowWaterMark, size_t* highWaterMark ) {
        return g_mockOptix->deviceContextGetCacheDatabaseSizes( context, lowWaterMark, highWaterMark );
    };
#if OPTIX_VERSION >= 70700
    funcTable.optixModuleCreate = []( OptixDeviceContext context, const OptixModuleCompileOptions* moduleCompileOptions,
                                      const OptixPipelineCompileOptions* pipelineCompileOptions, const char* input,
                                      size_t inputSize, char* logString, size_t* logStringSize, OptixModule* module ) {
        return g_mockOptix->moduleCreate( context, moduleCompileOptions, pipelineCompileOptions, input, inputSize,
                                          logString, logStringSize, module );
    };
#else
    funcTable.optixModuleCreateFromPTX = []( OptixDeviceContext context, const OptixModuleCompileOptions* moduleCompileOptions,
                                             const OptixPipelineCompileOptions* pipelineCompileOptions, const char* PTX,
                                             size_t PTXsize, char* logString, size_t* logStringSize, OptixModule* module ) {
        return g_mockOptix->moduleCreateFromPTX( context, moduleCompileOptions, pipelineCompileOptions, PTX, PTXsize,
                                                 logString, logStringSize, module );
    };
#endif
#if OPTIX_VERSION >= 70400
#if OPTIX_VERSION >= 70700
    funcTable.optixModuleCreateWithTasks = []( OptixDeviceContext context, const OptixModuleCompileOptions* moduleCompileOptions,
                                               const OptixPipelineCompileOptions* pipelineCompileOptions,
                                               const char* input, size_t inputSize, char* logString,
                                               size_t* logStringSize, OptixModule* module, OptixTask* firstTask ) {
        return g_mockOptix->moduleCreateWithTasks( context, moduleCompileOptions, pipelineCompileOptions, input,
                                                   inputSize, logString, logStringSize, module, firstTask );
    };
#else
    funcTable.optixModuleCreateFromPTXWithTasks =
        []( OptixDeviceContext context, const OptixModuleCompileOptions* moduleCompileOptions,
            const OptixPipelineCompileOptions* pipelineCompileOptions, const char* PTX, size_t PTXsize, char* logString,
            size_t* logStringSize, OptixModule* module, OptixTask* firstTask ) {
            return g_mockOptix->moduleCreateFromPTXWithTasks( context, moduleCompileOptions, pipelineCompileOptions, PTX,
                                                              PTXsize, logString, logStringSize, module, firstTask );
        };
#endif
    funcTable.optixModuleGetCompilationState = []( OptixModule module, OptixModuleCompileState* state ) {
        return g_mockOptix->moduleGetCompilationState( module, state );
    };
#endif
    funcTable.optixModuleDestroy = []( OptixModule module ) { return g_mockOptix->moduleDestroy( module ); };
    funcTable.optixBuiltinISModuleGet = []( OptixDeviceContext context, const OptixModuleCompileOptions* moduleCompileOptions,
                                            const OptixPipelineCompileOptions* pipelineCompileOptions,
                                            const OptixBuiltinISOptions* builtinISOptions, OptixModule* builtinModule ) {
        return g_mockOptix->builtinISModuleGet( context, moduleCompileOptions, pipelineCompileOptions, builtinISOptions, builtinModule );
    };
#if OPTIX_VERSION >= 70400
    funcTable.optixTaskExecute = []( OptixTask task, OptixTask* additionalTasks, unsigned int maxNumAdditionalTasks,
                                     unsigned int* numAdditionalTasksCreated ) {
        return g_mockOptix->taskExecute( task, additionalTasks, maxNumAdditionalTasks, numAdditionalTasksCreated );
    };
#endif
    funcTable.optixProgramGroupCreate = []( OptixDeviceContext context, const OptixProgramGroupDesc* programDescriptions,
                                            unsigned int numProgramGroups, const OptixProgramGroupOptions* options,
                                            char* logString, size_t* logStringSize, OptixProgramGroup* programGroups ) {
        return g_mockOptix->programGroupCreate( context, programDescriptions, numProgramGroups, options, logString,
                                                logStringSize, programGroups );
    };
    funcTable.optixProgramGroupDestroy = []( OptixProgramGroup group ) {
        return g_mockOptix->programGroupDestroy( group );
    };
#if OPTIX_VERSION >= 70700
    funcTable.optixProgramGroupGetStackSize = []( OptixProgramGroup programGroup, OptixStackSizes* stackSizes, OptixPipeline pipeline ) {
        return g_mockOptix->programGroupGetStackSize( programGroup, stackSizes, pipeline );
    };
#else
    funcTable.optixProgramGroupGetStackSize = []( OptixProgramGroup programGroup, OptixStackSizes* stackSizes ) {
        return g_mockOptix->programGroupGetStackSize( programGroup, stackSizes );
    };
#endif
    funcTable.optixPipelineCreate = []( OptixDeviceContext context, const OptixPipelineCompileOptions* pipelineCompileOptions,
                                        const OptixPipelineLinkOptions* pipelineLinkOptions,
                                        const OptixProgramGroup* programGroups, unsigned int numProgramGroups,
                                        char* logString, size_t* logStringSize, OptixPipeline* pipeline ) {
        return g_mockOptix->pipelineCreate( context, pipelineCompileOptions, pipelineLinkOptions, programGroups,
                                            numProgramGroups, logString, logStringSize, pipeline );
    };
    funcTable.optixPipelineDestroy = []( OptixPipeline pipeline ) { return g_mockOptix->pipelineDestroy( pipeline ); };
    funcTable.optixPipelineSetStackSize = []( OptixPipeline pipeline, unsigned int directCallableStackSizeFromTraversal,
                                              unsigned int directCallableStackSizeFromState,
                                              unsigned int continuationStackSize, unsigned int maxTraversableGraphDepth ) {
        return g_mockOptix->pipelineSetStackSize( pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
                                                  continuationStackSize, maxTraversableGraphDepth );
    };
    funcTable.optixAccelComputeMemoryUsage = []( OptixDeviceContext context, const OptixAccelBuildOptions* accelOptions,
                                                 const OptixBuildInput* buildInputs, unsigned int numBuildInputs,
                                                 OptixAccelBufferSizes* bufferSizes ) {
        return g_mockOptix->accelComputeMemoryUsage( context, accelOptions, buildInputs, numBuildInputs, bufferSizes );
    };
    funcTable.optixAccelBuild = []( OptixDeviceContext context, CUstream stream, const OptixAccelBuildOptions* accelOptions,
                                    const OptixBuildInput* buildInputs, unsigned int numBuildInputs,
                                    CUdeviceptr tempBuffer, size_t tempBufferSizeInBytes, CUdeviceptr outputBuffer,
                                    size_t outputBufferSizeInBytes, OptixTraversableHandle* outputHandle,
                                    const OptixAccelEmitDesc* emittedProperties, unsigned int numEmittedProperties ) {
        return g_mockOptix->accelBuild( context, stream, accelOptions, buildInputs, numBuildInputs, tempBuffer,
                                        tempBufferSizeInBytes, outputBuffer, outputBufferSizeInBytes, outputHandle,
                                        emittedProperties, numEmittedProperties );
    };
#if OPTIX_VERSION >= 70600
    funcTable.optixAccelGetRelocationInfo = []( OptixDeviceContext context, OptixTraversableHandle handle, OptixRelocationInfo* info ) {
        return g_mockOptix->accelGetRelocationInfo( context, handle, info );
    };
    funcTable.optixCheckRelocationCompatibility = []( OptixDeviceContext context, const OptixRelocationInfo* info, int* compatible ) {
        return g_mockOptix->checkRelocationCompatibility( context, info, compatible );
    };
    funcTable.optixAccelRelocate = []( OptixDeviceContext context, CUstream stream, const OptixRelocationInfo* info,
                                       const OptixRelocateInput* relocateInputs, size_t numRelocateInputs, CUdeviceptr targetAccel,
                                       size_t targetAccelSizeInBytes, OptixTraversableHandle* targetHandle ) {
        return g_mockOptix->accelRelocate( context, stream, info, relocateInputs, numRelocateInputs, targetAccel,
                                           targetAccelSizeInBytes, targetHandle );
    };
#else
    funcTable.optixAccelGetRelocationInfo = []( OptixDeviceContext context, OptixTraversableHandle handle,
                                                OptixAccelRelocationInfo* info ) {
        return g_mockOptix->accelGetRelocationInfo( context, handle, info );
    };
    funcTable.optixAccelCheckRelocationCompatibility = []( OptixDeviceContext              context,
                                                           const OptixAccelRelocationInfo* info, int* compatible ) {
        return g_mockOptix->accelCheckRelocationCompatibility( context, info, compatible );
    };
    funcTable.optixAccelRelocate = []( OptixDeviceContext context, CUstream stream, const OptixAccelRelocationInfo* info,
                                       CUdeviceptr instanceTraversablehandles, size_t numInstanceTraversableHandles,
                                       CUdeviceptr targetAccel, size_t targetAccelSizeInBytes, OptixTraversableHandle* targetHandle ) {
        return g_mockOptix->accelRelocate( context, stream, info, instanceTraversablehandles, numInstanceTraversableHandles,
                                           targetAccel, targetAccelSizeInBytes, targetHandle );
    };
#endif
    funcTable.optixAccelCompact = []( OptixDeviceContext context, CUstream stream, OptixTraversableHandle inputHandle,
                                      CUdeviceptr outputBuffer, size_t outputBufferSizeInBytes,
                                      OptixTraversableHandle* outputHandle ) {
        return g_mockOptix->accelCompact( context, stream, inputHandle, outputBuffer, outputBufferSizeInBytes, outputHandle );
    };
#if OPTIX_VERSION >= 70700
    funcTable.optixAccelEmitProperty = []( OptixDeviceContext context, CUstream stream, OptixTraversableHandle handle,
                                           const OptixAccelEmitDesc* emittedProperty ) {
        return g_mockOptix->accelEmitProperty( context, stream, handle, emittedProperty );
    };
#endif
    funcTable.optixConvertPointerToTraversableHandle = []( OptixDeviceContext onDevice, CUdeviceptr pointer,
                                                           OptixTraversableType    traversableType,
                                                           OptixTraversableHandle* traversableHandle ) {
        return g_mockOptix->convertPointerToTraversableHandle( onDevice, pointer, traversableType, traversableHandle );
    };
#if OPTIX_VERSION >= 70600
    funcTable.optixOpacityMicromapArrayComputeMemoryUsage = []( OptixDeviceContext                         context,
                                                                const OptixOpacityMicromapArrayBuildInput* buildInput,
                                                                OptixMicromapBufferSizes* bufferSizes ) {
        return g_mockOptix->opacityMicromapArrayComputeMemoryUsage( context, buildInput, bufferSizes );
    };
    funcTable.optixOpacityMicromapArrayBuild = []( OptixDeviceContext context, CUstream stream,
                                                   const OptixOpacityMicromapArrayBuildInput* buildInput,
                                                   const OptixMicromapBuffers*                buffers ) {
        return g_mockOptix->opacityMicromapArrayBuild( context, stream, buildInput, buffers );
    };
    funcTable.optixOpacityMicromapArrayGetRelocationInfo = []( OptixDeviceContext context, CUdeviceptr opacityMicromapArray,
                                                               OptixRelocationInfo* info ) {
        return g_mockOptix->opacityMicromapArrayGetRelocationInfo( context, opacityMicromapArray, info );
    };
    funcTable.optixOpacityMicromapArrayRelocate = []( OptixDeviceContext context, CUstream stream,
                                                      const OptixRelocationInfo* info, CUdeviceptr targetOpacityMicromapArray,
                                                      size_t targetOpacityMicromapArraySizeInBytes ) {
        return g_mockOptix->opacityMicromapArrayRelocate( context, stream, info, targetOpacityMicromapArray,
                                                          targetOpacityMicromapArraySizeInBytes );
    };
#endif
#if OPTIX_VERSION >= 70700
    funcTable.optixDisplacementMicromapArrayComputeMemoryUsage = []( OptixDeviceContext context,
                                                                     const OptixDisplacementMicromapArrayBuildInput* buildInput,
                                                                     OptixMicromapBufferSizes* bufferSizes ) {
        return g_mockOptix->displacementMicromapArrayComputeMemoryUsage( context, buildInput, bufferSizes );
    };
    funcTable.optixDisplacementMicromapArrayBuild = []( OptixDeviceContext context, CUstream stream,
                                                        const OptixDisplacementMicromapArrayBuildInput* buildInput,
                                                        const OptixMicromapBuffers*                     buffers ) {
        return g_mockOptix->displacementMicromapArrayBuild( context, stream, buildInput, buffers );
    };
#endif
    funcTable.optixSbtRecordPackHeader = []( OptixProgramGroup programGroup, void* sbtRecordHeaderHostPointer ) {
        return g_mockOptix->sbtRecordPackHeader( programGroup, sbtRecordHeaderHostPointer );
    };
    funcTable.optixLaunch = []( OptixPipeline pipeline, CUstream stream, CUdeviceptr pipelineParams, size_t pipelineParamsSize,
                                const OptixShaderBindingTable* sbt, unsigned int width, unsigned int height, unsigned int depth ) {
        return g_mockOptix->launch( pipeline, stream, pipelineParams, pipelineParamsSize, sbt, width, height, depth );
    };
    funcTable.optixDenoiserCreate = []( OptixDeviceContext context, OptixDenoiserModelKind modelKind,
                                        const OptixDenoiserOptions* options, OptixDenoiser* returnHandle ) {
        return g_mockOptix->denoiserCreate( context, modelKind, options, returnHandle );
    };
    funcTable.optixDenoiserDestroy = []( OptixDenoiser handle ) { return g_mockOptix->denoiserDestroy( handle ); };
    funcTable.optixDenoiserComputeMemoryResources = []( const OptixDenoiser handle, unsigned int maximumInputWidth,
                                                        unsigned int maximumInputHeight, OptixDenoiserSizes* returnSizes ) {
        return g_mockOptix->denoiserComputeMemoryResources( handle, maximumInputWidth, maximumInputHeight, returnSizes );
    };
    funcTable.optixDenoiserSetup = []( OptixDenoiser denoiser, CUstream stream, unsigned int inputWidth,
                                       unsigned int inputHeight, CUdeviceptr state, size_t stateSizeInBytes,
                                       CUdeviceptr scratch, size_t scratchSizeInBytes ) {
        return g_mockOptix->denoiserSetup( denoiser, stream, inputWidth, inputHeight, state, stateSizeInBytes, scratch,
                                           scratchSizeInBytes );
    };
    funcTable.optixDenoiserInvoke = []( OptixDenoiser denoiser, CUstream stream, const OptixDenoiserParams* params,
                                        CUdeviceptr denoiserState, size_t denoiserStateSizeInBytes,
                                        const OptixDenoiserGuideLayer* guideLayer, const OptixDenoiserLayer* layers,
                                        unsigned int numLayers, unsigned int inputOffsetX, unsigned int inputOffsetY,
                                        CUdeviceptr scratch, size_t scratchSizeInBytes ) {
        return g_mockOptix->denoiserInvoke( denoiser, stream, params, denoiserState, denoiserStateSizeInBytes, guideLayer,
                                            layers, numLayers, inputOffsetX, inputOffsetY, scratch, scratchSizeInBytes );
    };
    funcTable.optixDenoiserComputeIntensity = []( OptixDenoiser handle, CUstream stream, const OptixImage2D* inputImage,
                                                  CUdeviceptr outputIntensity, CUdeviceptr scratch, size_t scratchSizeInBytes ) {
        return g_mockOptix->denoiserComputeIntensity( handle, stream, inputImage, outputIntensity, scratch, scratchSizeInBytes );
    };
    funcTable.optixDenoiserComputeAverageColor = []( OptixDenoiser handle, CUstream stream,
                                                     const OptixImage2D* inputImage, CUdeviceptr outputAverageColor,
                                                     CUdeviceptr scratch, size_t scratchSizeInBytes ) {
        return g_mockOptix->denoiserComputeAverageColor( handle, stream, inputImage, outputAverageColor, scratch, scratchSizeInBytes );
    };
    funcTable.optixDenoiserCreateWithUserModel = []( OptixDeviceContext context, const void* data,
                                                     size_t dataSizeInBytes, OptixDenoiser* returnHandle ) {
        return g_mockOptix->denoiserCreateWithUserModel( context, data, dataSizeInBytes, returnHandle );
    };
}

}  // namespace testing
}  // namespace otk
