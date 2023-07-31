# Cuda Opacity Micromap Baking Library

The OptiX cuda Opacity Micromap Baking library allows hardware-accelerated baking of 
OptiX Opacity Micromap Arrays from textured geometry. 

It works by taking triangle texture coordinates and opacity textures as input, and generating all buffers needed to build the OptiX Opacity Micromap Array
and Geometry Acceleration Structures using the Opacity Micromap Array. For a higher-level discussion on OptiX Opacity Micromaps, please see the [OptiX Programming Guide]
(https://raytracing-docs.nvidia.com/optix7/guide/index.html#acceleration_structures#accelstruct-omm).

The library takes multiple inputs of textured geometry and generates data for a single shared OptiX Opacity Micromap Array.
The library scans the geometry to detect duplicate triangles, filters out triangles with uniform opacity, 
allocates available data memory over the remaining unique opacity micromaps proportional to their covered texel area, 
and evaluates the opacity states of all micro triangles in the opacity micromap array.
The main motivation to bake multiple inputs into a single opacity micromap array should be opacity micromap re-use.
If inputs do not share any opacity micromaps (for example because they use distinct textures) there is little benefit from
baking them together into a single opacity micromap array.
Micro triangle opacity states are conservative with respect to the underlying opacity texture. 
Micro triangles covering texels with varying opacity states evaluate to unknown opacity. 

The library does not allocate any device memory and relies on the user to allocate and pass in the output buffers. 
The library provides two functions, GetPreBakeInfo and BakeOpacityMicromaps.
The user queries the required output buffer sizes by calling GetPreBakeInfo. 
A following call to BakeOpacityMicromaps will launch the device tasks to generate the Opacity Micromap Array buffer contents.
All device tasks are launched asynchronously.

API documentation for the Cuda Opacticy Micromap Baking Library can be generated via `make docs` after configuring CMake.

## Quick start

See the [simple Opacity Micromap Baking example](../examples/CuOmmBaking/Simple/simple.cpp)
example, which demonstrates how to use the Cuda Opacticy Micromap Baking Library.

The first step is to setup a texture descriptor, specifying an opacity texture input.

```
// setup a texture input
TextureDesc texture = {};
texture.type = TextureType::CUDA;
texture.cuda.texObject = tex;
texture.cuda.transparencyCutoff = 0.f;
texture.cuda.opacityCutoff = 1.f;
```

Next, a bake input in created, specifying a textured triangle mesh.

```
// setup a bake input
BakeInputDesc bakeInput = {};
bakeInput.indexFormat = IndexFormat::I32_UINT;
bakeInput.indexBuffer = ( CUdeviceptr )indexBuffer;
bakeInput.numIndexTriplets = numIndexTriplets;

bakeInput.texCoordFormat = TexCoordFormat::UV32_FLOAT2;
bakeInput.texCoordBuffer = ( CUdeviceptr )texCoordBuffer;
bakeInput.numTexCoords = numTexCoords;

bakeInput.numTextures = 1;
bakeInput.textures = &texture;
```

Prior to baking, the `GetPreBakeInfo()` is called, which sets up the required sizes of all output buffers.

```
BakeOptions options = {};

// query the output buffer memory requirements
BakeInputBuffers inputBuffer;
BakeBuffers buffers;
GetPreBakeInfo( &options, 1, &bakeInput, &inputBuffer, &buffers );    
```

The requested output buffers are allocated.

```
size_t usageCountsSizeInBytes = inputBuffer.numMicromapUsageCounts * sizeof( OptixOpacityMicromapUsageCount );
size_t histogramSizeInBytes = buffers.numMicromapHistogramEntries * sizeof( OptixOpacityMicromapHistogramEntry );

cudaMalloc( ( void** )&inputBuffer.indexBuffer, inputBuffer.indexBufferSizeInBytes );
cudaMalloc( ( void** )&inputBuffer.micromapUsageCountsBuffer, usageCountsSizeInBytes );

cudaMalloc( ( void** )&buffers.outputBuffer, buffers.outputBufferSizeInBytes );
cudaMalloc( ( void** )&buffers.perMicromapDescBuffer, buffers.numMicromapDescs * sizeof( OptixOpacityMicromapDesc ) );
cudaMalloc( ( void** )&buffers.micromapHistogramEntriesBuffer, histogramSizeInBytes );
cudaMalloc( ( void** )&buffers.tmpBuffer, buffers.tmpBufferSizeInBytes );
```

The baking task is launched by calling `BakeOpacityMicromaps()`.

```
// launch baking
BakeOpacityMicromaps( &options, 1, &bakeInput, &inputBuffer, &buffers, stream );
```

The generated histogram and usage counts buffers are needed on the host as inputs to `optixOpacityMicromapArrayBuild` and `optixAccelBuild` respectively.
Thus, after the baking task completed the histogram and usage buffers are downloaded to the host.

```
std::vector<OptixOpacityMicromapHistogramEntry> histogram;
std::vector<OptixOpacityMicromapUsageCount> usageCounts;

histogram.resize(buffers.numMicromapHistogramEntries);
usageCounts.resize(inputBuffer.numMicromapUsageCounts);

cudaMemcpy( histogram.data(), buffers.micromapHistogramEntriesBuffer, histogramSizeInBytes, cudaMemcpyDeviceToHost );
cudaMemcpy( usageCounts.data(), inputBuffer.micromapUsageCountsBuffer, usageCountsSizeInBytes, cudaMemcpyDeviceToHost );
```
