# OptiX Demand Loading Geometry Library

The demand loading geometry library consists of a host API and a device API.
A simple example is provided that demonstrates the use of the API.

## Example

The `DemandGeometryViewer` example provides a simple demonstration of
demand loading geometry with a single proxy that is replaced with an instance of
a sphere primitive.

## Host API

The host API is accessed via the ProxyInstances class declared in
`<OptiXToolkit/DemandGeometry/ProxyInstances.h>`.  An instance of this class
is constructed from an existing instance of the `demandLoading::DemandLoader`
class.

The rendering process follows this sequence:
1. The application calls `ProxyInstances::add` to add proxies to the scene.
Proxies are specified by their associated axis-aligned bounding box (AABB).
Each added proxy returns an id that is associated with the proxy.  Proxies
can be removed from the scene by calling `ProxyInstances::remove` and supplying
their associated id.
2. Once the set of proxies has been updated for this launch,
`ProxyInstances::copyToDevice` or `ProxyInstances::copyToDeviceAsync` is called
to update the associated proxy data on the device.
3. Call `DemandLoader::launchPrepare` to prepare the demand loader for a launch.
4. Call `ProxyInstances::getContext` to obtain a `::demandGeometry::Context` structure
and copy that to the device.  Usually this will be a member of the launch parameters
structure, but it may reside anywhere in device memory.
5. Update any other application related state (e.g. launch parameters).
6. Call `optixLaunch` to traverse the scene.  Intersected proxies will have a
closest hit program that will report their intersection with generated rays.
7. Call `DemandLoader::processRequests` to process and demand-loaded textures
and get intersected proxies reported back to `ProxyInstances`.  This returns
a ticket to synchronize with the demand loader.
8. Call `Ticket::wait` on the returned ticket to get all proxy ids reported
back to `ProxyInstances`.
9. Call `ProxyInstances::requestedProxyIds` to get the ids of the intersected
proxies.  For each id, the application can either ignore the request or load
the corresponding real geometry and remove the proxy associated with the id.
Any proxy not removed from the scene may continue to be intersected and report
back its associated id.

The proxies are represented in the scene by a traversable handle obtained by
calling `ProxyInstances::createTraversable`.  This traversable handle is typically
referenced by an instance acceleration structure in the application's scene.
The application should recreate the traversable whenever the set of proxies
changes.  The proxy instances traversable is associated with an SBT hit group
record.  Call `ProxyInstances::setSbtIndex` before creating the proxy traversable
to indicate which shader binding table index to use; the default is zero.

The intersection and closest hit programs for the proxy instances need to be
included in an `OptixProgramGroup`.  Since the implementation is included as
source in the application's CUDA code, there is no separate `OptixModule` for
the proxy instance programs.  The symbol names of the entry points to the programs
is obtained by calling `ProxyInstances::getCHFunctionName` and
`ProxyInstances::getISFunctionName`.  The hit group associated with these programs
is designated by the value passed to `ProxyInstances::setSbtIndex`.

## Device API

The proxy instances have their own intersection and closest hit programs.  The
application obtains the implementaiton of these programs by including them directly into
their CUDA source by including the file `<OptiXToolkit/DemandGeometry/ProxyInstancesImpl.h>`.
The application must implement 3 device-side functions called by the proxy instance
programs:

1. `::demandGeometry::app::getContext` returns a reference to the `::demandGeometry::Context`
structure copied to the device, typically a member of the launch parameters structure.
2. `::demandGeometry::app::getDeviceContext` returns a reference to the
`::demandLoading::DeviceContext` structure, typically a member of the launch
parameters structure.
3. `::demandGeometry::app::reportClosestHitNormal` reports the normal of the intersected
proxy.  The application can visualize intersected proxies by using the associated normal
to report a color on the ray payload.  If no visual representation of the proxy is desired,
then implement an empty function.
