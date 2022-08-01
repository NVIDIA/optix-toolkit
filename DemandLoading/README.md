# OptiX Demand Loading Library

The OptiX Demand Loading library allows hardware-accelerated sparse textures to be loaded on demand,
which greatly reduces memory requirements, bandwidth, and disk I/O compared to preloading
textures. It works by maintaining a page table that tracks which texture tiles have been loaded into
GPU memory. An OptiX closest-hit program fetches from a texture by calling library device code that
checks the page table to see if the required tiles are present. If not, the library records a page
request, which is processed by library host code after the kernel has terminated. Kernel launches
are repeated until no tiles are missing, typically using adaptive sampling to avoid redundant
work. Although it is currently focused on texturing, much of the library is generic and can be
adapted to load arbitrary data on demand, such as per-vertex data like colors or normals.

A quick start guide is provided below.  For additional information, please see the [OptiX Programming Guide Guide](https://raytracing-docs.nvidia.com/optix7/guide/index.html#demand_loading#demand-loaded-sparse-textures) (section 15).

API documentation for the Demand Loading library can be generated via `make docs` after configuring CMake.

## Quick start

See the [simple demand loading example](../examples/DemandLoading/Simple/simple.cpp)
example, which demonstrates how to use the OptiX Demand Loading library.  The first step is to
create a `DemandLoader` object, through which subsequent method calls are invoked:
```
// Create DemandLoader
DemandLoader* loader = createDemandLoader( Options() );
```
Next, a demand-loaded resource is created, specifying a callback to handle page requests.  The
nature of the resource is entirely user-programmable.  For example, a resource might represent a
large buffer of per-vertex data like normals, from which pages might be loaded on demand.
```
// Create a resource, using the given callback to handle page requests.
const unsigned int numPages  = 128;
unsigned int       startPage = loader->createResource( numPages, callback );
```
Prior to launching a kernel, the `launchPrepare()` method is called, which returns a `DeviceContext` structure via a result parameter:
```
// Prepare for launch, obtaining DeviceContext.
DeviceContext context;
loader->launchPrepare( deviceIndex, stream, context );
```
The `DeviceContext` structure should be passed to the kernel as a launch parameter, which copies it
to device memory.  In the [kernel device code](../examples/optixDemandLoadSimple/PageRequester.cu), the
`pagingMapOrRequest()` function can be used to request a page of data.  It returns a pointer to the
requested data (or a null pointer if it's not resident), along with a boolean result parameter
indicating whether the requested page is resident.
```
bool isResident;
void* data = pagingMapOrRequest( context, pageId, &isResident );
```
After the kernel has been launched, the `processRequests()` method is used to initiate processing of any pages requested by the
kernel.  In normal usage, the same CUDA stream should be used for `launchPrepare()`, the kernel launch, and `processRequests()`.
```
// Initiate request processing, which returns a Ticket.
Ticket ticket = loader->processRequests( deviceIndex, stream, context );
```
Request processing is asynchronous (and so is the kernel launch), so the `processRequests()` method returns a ticket
that can be used to wait until all the page requests have been processed.
```
// Wait for any page requests to be processed.
ticket.wait();
```
The kernel is then relaunched in a multi-pass fashion until all the requests have been processed.  This process can be terminated by
checking for an empty `Ticket`:
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
