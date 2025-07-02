// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <optix.h>
#include <optix_function_table.h>

#ifndef OPTIX_FUNCTION_TABLE_SYMBOL
#define OPTIX_FUNCTION_TABLE_SYMBOL g_optixFunctionTable  // pre-OptiX 8.1
#endif

#include <gmock/gmock.h>

namespace otk {
namespace testing {

/// MockOptix
///
/// This is a mock class that implements the OptiX API versions 7.3 through 7.7.
/// The global API functions delegate through the OptixFunctionTable instance, a
/// struct of function pointers.  The function table is initialized with functions
/// that delegate to an instance of this mock class.  The names of the members of
/// this mock class are the same as the OptiX global API function calls with the
/// "optix" prefix removed and the first letter of the next word in the function
/// name changed to lower case.  For instance the API function optixGetErrorName
/// has a mock method MockOptix::getErrorName.
///
/// The global instance of the mock class is initialized with a call to
/// otk::testing::initMockOptix.
///
/// NOTE: When using this in test applications, no other source files in your
/// test executable should include <optix_function_table_definition.h> or define
/// g_optixFunctionTable.
///
/// ***This includes any application code that links against the test executable.***
///
/// The easiest way to arrange this scenario is to isolate application code to be
/// tested in a library that does not define the global function table and define
/// the function table in some other library or wherever the main entry point of
/// the application is defined.
///
class MockOptix
{
  public:
    MockOptix();
    virtual ~MockOptix();

    MOCK_METHOD( const char*, getErrorName, ( OptixResult result ) );
    MOCK_METHOD( const char*, getErrorString, ( OptixResult result ) );
    MOCK_METHOD( OptixResult,
                 deviceContextCreate,
                 ( CUcontext fromContext, const OptixDeviceContextOptions* options, OptixDeviceContext* context ) );
    MOCK_METHOD( OptixResult, deviceContextDestroy, ( OptixDeviceContext context ) );
    MOCK_METHOD( OptixResult,
                 deviceContextGetProperty,
                 ( OptixDeviceContext context, OptixDeviceProperty property, void* value, size_t sizeInBytes ) );
    MOCK_METHOD( OptixResult,
                 deviceContextSetLogCallback,
                 ( OptixDeviceContext context, OptixLogCallback callbackFunction, void* callbackData, unsigned int callbackLevel ) );
    MOCK_METHOD( OptixResult, deviceContextSetCacheEnabled, ( OptixDeviceContext context, int enabled ) );
    MOCK_METHOD( OptixResult, deviceContextSetCacheLocation, ( OptixDeviceContext context, const char* location ) );
    MOCK_METHOD( OptixResult, deviceContextSetCacheDatabaseSizes, ( OptixDeviceContext context, size_t lowWaterMark, size_t highWaterMark ) );
    MOCK_METHOD( OptixResult, deviceContextGetCacheEnabled, ( OptixDeviceContext context, int* enabled ) );
    MOCK_METHOD( OptixResult, deviceContextGetCacheLocation, ( OptixDeviceContext context, char* location, size_t locationSize ) );
    MOCK_METHOD( OptixResult, deviceContextGetCacheDatabaseSizes, ( OptixDeviceContext context, size_t* lowWaterMark, size_t* highWaterMark ) );
#if OPTIX_VERSION >= 70700
    MOCK_METHOD( OptixResult,
                 moduleCreate,
                 ( OptixDeviceContext                 context,
                   const OptixModuleCompileOptions*   moduleCompileOptions,
                   const OptixPipelineCompileOptions* pipelineCompileOptions,
                   const char*                        input,
                   size_t                             inputSize,
                   char*                              logString,
                   size_t*                            logStringSize,
                   OptixModule*                       module ) );
    MOCK_METHOD( OptixResult,
                 moduleCreateWithTasks,
                 ( OptixDeviceContext                 context,
                   const OptixModuleCompileOptions*   moduleCompileOptions,
                   const OptixPipelineCompileOptions* pipelineCompileOptions,
                   const char*                        input,
                   size_t                             inputSize,
                   char*                              logString,
                   size_t*                            logStringSize,
                   OptixModule*                       module,
                   OptixTask*                         firstTask ) );
#else
    MOCK_METHOD( OptixResult,
                 moduleCreateFromPTX,
                 ( OptixDeviceContext                 context,
                   const OptixModuleCompileOptions*   moduleCompileOptions,
                   const OptixPipelineCompileOptions* pipelineCompileOptions,
                   const char*                        PTX,
                   size_t                             PTXsize,
                   char*                              logString,
                   size_t*                            logStringSize,
                   OptixModule*                       module ) );
#if OPTIX_VERSION >= 70400
    MOCK_METHOD( OptixResult,
                 moduleCreateFromPTXWithTasks,
                 ( OptixDeviceContext                 context,
                   const OptixModuleCompileOptions*   moduleCompileOptions,
                   const OptixPipelineCompileOptions* pipelineCompileOptions,
                   const char*                        PTX,
                   size_t                             PTXsize,
                   char*                              logString,
                   size_t*                            logStringSize,
                   OptixModule*                       module,
                   OptixTask*                         firstTask ) );
#endif
#endif
#if OPTIX_VERSION >= 70400
    MOCK_METHOD( OptixResult, moduleGetCompilationState, ( OptixModule module, OptixModuleCompileState* state ) );
#endif
    MOCK_METHOD( OptixResult, moduleDestroy, ( OptixModule module ) );
    MOCK_METHOD( OptixResult,
                 builtinISModuleGet,
                 ( OptixDeviceContext                 context,
                   const OptixModuleCompileOptions*   moduleCompileOptions,
                   const OptixPipelineCompileOptions* pipelineCompileOptions,
                   const OptixBuiltinISOptions*       builtinISOptions,
                   OptixModule*                       builtinModule ) );
#if OPTIX_VERSION >= 70400
    MOCK_METHOD( OptixResult,
                 taskExecute,
                 ( OptixTask task, OptixTask* additionalTasks, unsigned int maxNumAdditionalTasks, unsigned int* numAdditionalTasksCreated ) );
#endif
    MOCK_METHOD( OptixResult,
                 programGroupCreate,
                 ( OptixDeviceContext              context,
                   const OptixProgramGroupDesc*    programDescriptions,
                   unsigned int                    numProgramGroups,
                   const OptixProgramGroupOptions* options,
                   char*                           logString,
                   size_t*                         logStringSize,
                   OptixProgramGroup*              programGroups ) );
    MOCK_METHOD( OptixResult, programGroupDestroy, ( OptixProgramGroup group ) );
#if OPTIX_VERSION >= 70700
    MOCK_METHOD( OptixResult,
                 programGroupGetStackSize,
                 ( OptixProgramGroup programGroup, OptixStackSizes* stackSizes, OptixPipeline pipeline ) );
#else
    MOCK_METHOD( OptixResult, programGroupGetStackSize, ( OptixProgramGroup programGroup, OptixStackSizes* stackSizes ) );
#endif
    MOCK_METHOD( OptixResult,
                 pipelineCreate,
                 ( OptixDeviceContext                 context,
                   const OptixPipelineCompileOptions* pipelineCompileOptions,
                   const OptixPipelineLinkOptions*    pipelineLinkOptions,
                   const OptixProgramGroup*           programGroups,
                   unsigned int                       numProgramGroups,
                   char*                              logString,
                   size_t*                            logStringSize,
                   OptixPipeline*                     pipeline ) );
    MOCK_METHOD( OptixResult, pipelineDestroy, ( OptixPipeline pipeline ) );
    MOCK_METHOD( OptixResult,
                 pipelineSetStackSize,
                 ( OptixPipeline pipeline,
                   unsigned int  directCallableStackSizeFromTraversal,
                   unsigned int  directCallableStackSizeFromState,
                   unsigned int  continuationStackSize,
                   unsigned int  maxTraversableGraphDepth ) );
    MOCK_METHOD( OptixResult,
                 accelComputeMemoryUsage,
                 ( OptixDeviceContext            context,
                   const OptixAccelBuildOptions* accelOptions,
                   const OptixBuildInput*        buildInputs,
                   unsigned int                  numBuildInputs,
                   OptixAccelBufferSizes*        bufferSizes ) );
    MOCK_METHOD( OptixResult,
                 accelBuild,
                 ( OptixDeviceContext            context,
                   CUstream                      stream,
                   const OptixAccelBuildOptions* accelOptions,
                   const OptixBuildInput*        buildInputs,
                   unsigned int                  numBuildInputs,
                   CUdeviceptr                   tempBuffer,
                   size_t                        tempBufferSizeInBytes,
                   CUdeviceptr                   outputBuffer,
                   size_t                        outputBufferSizeInBytes,
                   OptixTraversableHandle*       outputHandle,
                   const OptixAccelEmitDesc*     emittedProperties,
                   unsigned int                  numEmittedProperties ) );
#if OPTIX_VERSION >= 70600
    MOCK_METHOD( OptixResult,
                 accelGetRelocationInfo,
                 ( OptixDeviceContext context, OptixTraversableHandle handle, OptixRelocationInfo* info ) );
    MOCK_METHOD( OptixResult,
                 checkRelocationCompatibility,
                 ( OptixDeviceContext context, const OptixRelocationInfo* info, int* compatible ) );
    MOCK_METHOD( OptixResult,
                 accelRelocate,
                 ( OptixDeviceContext         context,
                   CUstream                   stream,
                   const OptixRelocationInfo* info,
                   const OptixRelocateInput*  relocateInputs,
                   size_t                     numRelocateInputs,
                   CUdeviceptr                targetAccel,
                   size_t                     targetAccelSizeInBytes,
                   OptixTraversableHandle*    targetHandle ) );
#else
    MOCK_METHOD( OptixResult,
                 accelGetRelocationInfo,
                 ( OptixDeviceContext context, OptixTraversableHandle handle, OptixAccelRelocationInfo* info ) );
    MOCK_METHOD( OptixResult,
                 accelCheckRelocationCompatibility,
                 ( OptixDeviceContext context, const OptixAccelRelocationInfo* info, int* compatible ) );
    MOCK_METHOD( OptixResult,
                 accelRelocate,
                 ( OptixDeviceContext              context,
                   CUstream                        stream,
                   const OptixAccelRelocationInfo* info,
                   CUdeviceptr                     instanceTraversablehandles,
                   size_t                          numInstanceTraversableHandles,
                   CUdeviceptr                     targetAccel,
                   size_t                          targetAccelSizeInBytes,
                   OptixTraversableHandle*         targetHandle ) );
#endif
    MOCK_METHOD( OptixResult,
                 accelCompact,
                 ( OptixDeviceContext      context,
                   CUstream                stream,
                   OptixTraversableHandle  inputHandle,
                   CUdeviceptr             outputBuffer,
                   size_t                  outputBufferSizeInBytes,
                   OptixTraversableHandle* outputHandle ) );
#if OPTIX_VERSION >= 70700
    MOCK_METHOD( OptixResult,
                 accelEmitProperty,
                 ( OptixDeviceContext context, CUstream stream, OptixTraversableHandle handle, const OptixAccelEmitDesc* emittedProperty ) );
#endif
    MOCK_METHOD( OptixResult,
                 convertPointerToTraversableHandle,
                 ( OptixDeviceContext onDevice, CUdeviceptr pointer, OptixTraversableType traversableType, OptixTraversableHandle* traversableHandle ) );
#if OPTIX_VERSION >= 70600
    MOCK_METHOD( OptixResult,
                 opacityMicromapArrayComputeMemoryUsage,
                 ( OptixDeviceContext context, const OptixOpacityMicromapArrayBuildInput* buildInput, OptixMicromapBufferSizes* bufferSizes ) );
    MOCK_METHOD( OptixResult,
                 opacityMicromapArrayBuild,
                 ( OptixDeviceContext                         context,
                   CUstream                                   stream,
                   const OptixOpacityMicromapArrayBuildInput* buildInput,
                   const OptixMicromapBuffers*                buffers ) );
    MOCK_METHOD( OptixResult,
                 opacityMicromapArrayGetRelocationInfo,
                 ( OptixDeviceContext context, CUdeviceptr opacityMicromapArray, OptixRelocationInfo* info ) );
    MOCK_METHOD( OptixResult,
                 opacityMicromapArrayRelocate,
                 ( OptixDeviceContext         context,
                   CUstream                   stream,
                   const OptixRelocationInfo* info,
                   CUdeviceptr                targetOpacityMicromapArray,
                   size_t                     targetOpacityMicromapArraySizeInBytes ) );
#endif
#if OPTIX_VERSION >= 70700
    MOCK_METHOD( OptixResult,
                 displacementMicromapArrayComputeMemoryUsage,
                 ( OptixDeviceContext context, const OptixDisplacementMicromapArrayBuildInput* buildInput, OptixMicromapBufferSizes* bufferSizes ) );
    MOCK_METHOD( OptixResult,
                 displacementMicromapArrayBuild,
                 ( OptixDeviceContext                              context,
                   CUstream                                        stream,
                   const OptixDisplacementMicromapArrayBuildInput* buildInput,
                   const OptixMicromapBuffers*                     buffers ) );
#endif
    MOCK_METHOD( OptixResult, sbtRecordPackHeader, ( OptixProgramGroup programGroup, void* sbtRecordHeaderHostPointer ) );
    MOCK_METHOD( OptixResult,
                 launch,
                 ( OptixPipeline                  pipeline,
                   CUstream                       stream,
                   CUdeviceptr                    pipelineParams,
                   size_t                         pipelineParamsSize,
                   const OptixShaderBindingTable* sbt,
                   unsigned int                   width,
                   unsigned int                   height,
                   unsigned int                   depth ) );
    MOCK_METHOD( OptixResult,
                 denoiserCreate,
                 ( OptixDeviceContext context, OptixDenoiserModelKind modelKind, const OptixDenoiserOptions* options, OptixDenoiser* returnHandle ) );
    MOCK_METHOD( OptixResult, denoiserDestroy, ( OptixDenoiser handle ) );
    MOCK_METHOD( OptixResult,
                 denoiserComputeMemoryResources,
                 ( const OptixDenoiser handle, unsigned int maximumInputWidth, unsigned int maximumInputHeight, OptixDenoiserSizes* returnSizes ) );
    MOCK_METHOD( OptixResult,
                 denoiserSetup,
                 ( OptixDenoiser denoiser,
                   CUstream      stream,
                   unsigned int  inputWidth,
                   unsigned int  inputHeight,
                   CUdeviceptr   state,
                   size_t        stateSizeInBytes,
                   CUdeviceptr   scratch,
                   size_t        scratchSizeInBytes ) );
    MOCK_METHOD( OptixResult,
                 denoiserInvoke,
                 ( OptixDenoiser                  denoiser,
                   CUstream                       stream,
                   const OptixDenoiserParams*     params,
                   CUdeviceptr                    denoiserState,
                   size_t                         denoiserStateSizeInBytes,
                   const OptixDenoiserGuideLayer* guideLayer,
                   const OptixDenoiserLayer*      layers,
                   unsigned int                   numLayers,
                   unsigned int                   inputOffsetX,
                   unsigned int                   inputOffsetY,
                   CUdeviceptr                    scratch,
                   size_t                         scratchSizeInBytes ) );
    MOCK_METHOD( OptixResult,
                 denoiserComputeIntensity,
                 ( OptixDenoiser handle, CUstream stream, const OptixImage2D* inputImage, CUdeviceptr outputIntensity, CUdeviceptr scratch, size_t scratchSizeInBytes ) );
    MOCK_METHOD( OptixResult,
                 denoiserComputeAverageColor,
                 ( OptixDenoiser handle, CUstream stream, const OptixImage2D* inputImage, CUdeviceptr outputAverageColor, CUdeviceptr scratch, size_t scratchSizeInBytes ) );
    MOCK_METHOD( OptixResult,
                 denoiserCreateWithUserModel,
                 ( OptixDeviceContext context, const void* data, size_t dataSizeInBytes, OptixDenoiser* returnHandle ) );
};

void initMockOptix( MockOptix& mock );

}  // namespace testing
}  // namespace otk
