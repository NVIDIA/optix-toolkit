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

OptixFunctionTable g_optixFunctionTable{};

namespace otk {
namespace testing {

MockOptix* g_mockOptix{};

void initMockOptix( MockOptix& mock )
{
    g_mockOptix                                   = &mock;
    g_optixFunctionTable.optixDeviceContextCreate = []( CUcontext fromContext, const OptixDeviceContextOptions* options,
                                                        OptixDeviceContext* context ) {
        return g_mockOptix->deviceContextCreate( fromContext, options, context );
    };
    g_optixFunctionTable.optixDeviceContextDestroy = []( OptixDeviceContext context ) {
        return g_mockOptix->deviceContextDestroy( context );
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
    g_optixFunctionTable.optixModuleCreateFromPTX =
        []( OptixDeviceContext context, const OptixModuleCompileOptions* moduleCompileOptions,
            const OptixPipelineCompileOptions* pipelineCompileOptions, const char* PTX, size_t PTXsize, char* logString,
            size_t* logStringSize, OptixModule* module ) {
            return g_mockOptix->moduleCreateFromPTX( context, moduleCompileOptions, pipelineCompileOptions, PTX,
                                                     PTXsize, logString, logStringSize, module );
        };
    g_optixFunctionTable.optixModuleDestroy = []( OptixModule module ) { return g_mockOptix->moduleDestroy( module ); };
    g_optixFunctionTable.optixProgramGroupCreate =
        []( OptixDeviceContext context, const OptixProgramGroupDesc* programDescriptions, unsigned int numProgramGroups,
            const OptixProgramGroupOptions* options, char* logString, size_t* logStringSize, OptixProgramGroup* programGroups ) {
            return g_mockOptix->programGroupCreate( context, programDescriptions, numProgramGroups, options, logString,
                                                    logStringSize, programGroups );
        };
    g_optixFunctionTable.optixProgramGroupDestroy = []( OptixProgramGroup group ) {
        return g_mockOptix->programGroupDestroy( group );
    };
}

}  // namespace testing
}  // namespace otk
