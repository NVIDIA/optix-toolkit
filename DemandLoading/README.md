# OptiX Demand Loading Library

The OptiX Demand Loading library allows hardware-accelerated sparse textures, and other assets, to be loaded on demand based on requests initiated on the device in an OptiX program (or CUDA kernel). 

The demand loading library works by maintaining a page table that tracks assets, such as texture tiles or meshes, that can be requested on the GPU.  An OptiX program (raygen, closest-hit, etc.) requests an asset by calling a `mapOrRequest` function. This sets a request bit in the page table, and retrieves the asset if it is resident on the GPU. Otherwise, the program terminates the ray and waits for a successive OptiX kernel launch to complete the work that relied on the asset.  After the launch, the application pulls requests made by the kernel to the host and fills them.  Kernel launches are then repeated until all requests are resolved.

The OptiX Demand Loading library was initially designed for sparse textures, and that functionality is fairly mature now, but is supports other use cases as well, including on-demand geometry and material loading.  For more information on specific use cases of demand loading and supporting libraries please see the following:

- [Geometry](https://github.com/NVIDIA/optix-toolkit/tree/master/DemandLoading/DemandGeometry)
- [Materials](https://github.com/NVIDIA/optix-toolkit/tree/master/DemandLoading/DemandMaterial)
- [Sparse Textures](https://github.com/NVIDIA/otk-demand-loading/docs/README.md)

API documentation for the Demand Loading library can be generated via `make docs` after configuring CMake. A quick start guide for incorporating the Demand Loading Library into your projects is provided below.

## Quick Start Guide

The [simple demand loading example](../examples/DemandLoading/Simple/simple.cpp) demonstrates basic use of the OptiX Demand Loading library.  Following this example, first create a `DemandLoader` object, through which subsequent method calls will be invoked:
```
// Create DemandLoader
DemandLoader* loader = createDemandLoader( Options() );
```
Next, create a demand-loaded resource, specifying a callback to handle page requests.  The nature of the resource is entirely user-programmable.  For example, a resource might represent a large buffer of per-vertex data like normals, from which pages might be loaded on demand.
```
// Create a resource, using the given callback to handle page requests.
const unsigned int numPages  = 128;
unsigned int       startPage = loader->createResource( numPages, callback );
```
Prior to launching a kernel, call `launchPrepare()`, which returns a `DeviceContext` structure via a result parameter:
```
// Prepare for launch, obtaining DeviceContext.
DeviceContext context;
loader->launchPrepare( deviceIndex, stream, context );
```
Pass the `DeviceContext` structure to the OptiX kernel as a launch parameter, which copies it to device memory.  In the [kernel device code](../examples/optixDemandLoadSimple/PageRequester.cu), use the `pagingMapOrRequest()` function to request a page of data.  It returns a pointer to the requested data (or a null pointer if it's not resident), along with a boolean result parameter indicating whether the requested page is resident.
```
bool isResident;
void* data = pagingMapOrRequest( context, pageId, &isResident );
```
After the kernel has been launched, call `processRequests()` to initiate processing of any pages requested by the kernel.  In normal usage, the same CUDA stream should be used for `launchPrepare()`, the kernel launch, and `processRequests()`.
```
// Initiate request processing, which returns a Ticket.
Ticket ticket = loader->processRequests( deviceIndex, stream, context );
```
Request processing is asynchronous (and so is the kernel launch), so the `processRequests()` method returns a ticket that can be used to wait until all the page requests have been processed.
```
// Wait for any page requests to be processed.
ticket.wait();
```
Finally, relaunch the kernel in multi-pass fashion until all the requests have been processed, which can be determined by checking for an empty `Ticket`:
```
while( true )
{
    DeviceContext context;
    loader->launchPrepare( deviceIndex, stream, context );
    launchKernel( stream, context );
    Ticket ticket = loader->processRequests( deviceIndex, stream, context );
    ticket.wait();
    if( ticket.numTasksTotal() == 0 )
        break;
}
```
