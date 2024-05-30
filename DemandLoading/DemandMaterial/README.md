# OptiX Demand Loading Material Library

The demand loading material library consists of a host API and a device API.
A simple example is provided that demonstrates the use of the API.

## Example

The `DemandGeometryViewer` example provides a simple demonstration of
demand loading geometry with a single proxy that is replaced with an instance of
a sphere primitive using a proxy material.  The proxy material is then replaced
with a simple Phong material.

## Host API

The host API is accessed via the MaterialLoader interface declared in
`<OptiXToolkit/DemandMaterial/MaterialLoader.h>`.  An implementation of this interface
is constructed from an existing instance of the `demandLoading::DemandLoader`
class using the `createMaterialLoader` factory function.

The rendering process is similar to that used in the DemandGeometry library and
follows this sequence:
1. The application calls `MaterialLoader::add` to add a proxy material to the scene.
Proxies are specified by their associated axis-aligned bounding box (AABB).
Each added proxy returns an id that is associated with the proxy.
2. Update any other application related state (e.g. launch parameters).
3. Call `optixLaunch` to traverse the scene.  Intersected proxies will have a
closest hit program that will report their intersection with generated rays.
4. Call `DemandLoader::processRequests` to process and demand-loaded textures
and get intersected proxy materials reported back to `MaterialLoader`.  This
returns a ticket to synchronize with the demand loader.
8. Call `Ticket::wait` on the returned ticket to get all proxy ids reported
back to `MaterialLoader`.
9. Call `MaterialLoader::requestedMaterialIds` to get the ids of the intersected
proxy materials.  For each id, the application can either ignore the request or load
the corresponding real material and remove the proxy material associated with the id
by calling `MaterialLoader::remove`.  Any proxy not removed from the scene will
continue to be intersected and report back its associated id.

There is no device-side data for a proxy material, other than it's associated
paging system id.

The application associates the proxy material with application geometry by using
the proxy material's closest hit program in a hit group with the application's
intersection program.  The closest hit program for the proxy materials needs to be
included in an `OptixProgramGroup`.  Since the implementation is included as source
in the application's CUDA code, there is no separate `OptixModule` for the proxy
material closest hit program.  The symbol names of the entry points to the closest
hit program is obtained by calling `MaterialLoader::getCHFunctionName`.

The the proxy material closest hit program does not rely on any attributes from the
intersection program.

## Device API

The proxy materials have their own closest hit program.  The application obtains the
implementaiton of this program by including it directly into their CUDA source by
including the file `<OptiXToolkit/DemandMaterial/MaterialLoaderImpl.h>`.
The application must implement 3 device-side functions called by the proxy material
closest hit program:

1. `::demandMaterial::app::getDeviceContext` returns a reference to the
`::demandLoading::DeviceContext` structure, typically a member of the launch
parameters structure.
2. `::demandMaterial::app::getMaterialId` returns the proxy material id asssociated with
the intersected geometry.
3. `::demandMaterial::app::reportClosestHit` reports the material id of the intersected
proxy and a boolean indicating whether or not this material has been resolved.  The
application can visualize intersected proxy materials by using the associated id
to report a color on the ray payload.  If no visual representation of the proxy is desired,
then implement an empty function.
