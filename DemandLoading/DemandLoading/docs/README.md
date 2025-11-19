
# Demand-loaded sparse textures

The OptiX Demand Loading library allows hardware-accelerated sparse textures to be loaded on demand, which greatly reduces memory requirements, bandwidth, and disk I/O compared to preloading textures. It works by maintaining a page table that tracks which texture tiles have been loaded into GPU memory. An OptiX program fetches from a texture by calling library device code that checks the page table to see if the required tiles are present. If not, the library records a page request, which is processed by library host code after the kernel has terminated. Kernel launches are repeated until no tiles are missing, typically using adaptive sampling to avoid redundant work.

The full source for the OptiX Demand Loading library is provided in the OTK, along with samples that illustrates its use. It is free for commercial use. The source code is accompanied by documentation that describes the implementation in detail. Questions are welcome on the NVIDIA OptiX forums.

We expect that some developers will fork the library to customize it, while others will rely on NVIDIA for continued feature development.

Before describing the design and implementation of the library in more detail, we provide some motivation and background information.

## Motivation

The size of film-quality texture assets has been a longstanding obstacle to rendering final-frame images for visual effects shots and animated films, and this is especially true on GPUs which normally have less memory available to them than a CPU. It's not unusual for a hero character to have 50-100 GB of textures, and many sets and props are equally detailed (sometimes for no good reason, except that it would be more expensive to author at a more appropriate resolution).

Film models are usually painted at extremely high resolution in case of extreme closeup. A character's face is typically divided into several regions (for example, eye, nose, mouth), each of which is painted at 4K or 8K resolution. In addition, numerous layers may be used, including coarse and fine displacement and various material layers such as dirt, each of which is a separate collection of textures.

It's not practical to preload entire film-quality textures into GPU memory.
Besides wasting GPU memory, it also requires time and bandwidth to read the textures from disk and transfer them to GPU memory.  Nor is it possible to downscale the images, since the level of detail that is required depends on the distance of the textured object from the camera.  The ad-hoc solution of clamping the maximum resolution can lead to a loss of detail that is not acceptable in a film-quality renderer.

To make matters worse, it's not unusual for a production shot to have thousands of textured objects that are outside the camera view or completely occluded by other objects.  It's generally not possible to know *a priori* which textures are required without actually rendering a scene.

All of these factors make it difficult to load textures into GPU memory before rendering.  Our approach is a multi-pass one: the initial passes identify missing texture data, which is loaded from disk on demand and transferred to GPU memory between passes.

## Background

Mipmapping is a well known technique for dealing with high resolution textures. The idea is to precompute downscaled images, as illustrated below, that can be used when a lower level of detail is acceptable, such as when a textured object is far from the camera.

![Mipmapped texture showing downscaled images](images/demand_loading_1.png)

In a ray tracer, choosing which level of detail to use is usually accomplished using **ray differentials** to determine the gradients of the texture coordinates. The texture hardware on the GPU then uses those gradients to choose two miplevels, it performs a filtered lookup in each miplevel, and then blends between them.

Needless to say, that's an expensive operation. But it's necessary to avoid temporal aliasing when the camera or the geometry is moving, which would otherwise create a pop when the choice of miplevels suddenly changes from one frame to the next. GPU acceleration for mipmapped texturing is often taken for granted, and it offers a huge performance gain compared to texturing on the CPU.  Getting the texture data to the GPU and storing it there for rendering is the difficult part.

### Sparse textures

Mipmapped textures can be tiled, which divides each miplevel into a number of smaller sub-images. This allows a sparse memory layout to be used. Instead of loading an entire miplevel onto the GPU, only the individual tiles that are required are loaded, which saves memory and bandwidth.

Consider rendering a quad angled away from the camera, as shown below. The region that is closest to the camera must use tiles from a high resolution miplevel to ensure adequate detail. However, tiles from coarser miplevels can be used farther from the camera.

![Texture-mapped quad angled away from the camera](images/demand_loading_2.png)

The tiles highlighted below illustrate which tiles might be required for the rendering just shown:

![Tiles require](images/demand_loading_3.png)

Sparse textures are supported in CUDA, starting with the CUDA 11.1 toolkit. Under the hood, CUDA sparse textures employ the virtual memory system in a clever way. The main complication is that texture filtering often requires samples from multiple tiles for a single texture fetch. In that case, the GPU texture units require those tiles to be contiguous in memory. But it would be wasteful to reserve enough space to allow that.

The trick is to use the virtual memory system to provide the illusion of contiguous tiles to the GPU texture units. That's accomplished by binding the texture sampler to virtual memory, and then mapping individual tiles as virtual pages.

The following diagram illustrates how that works. On the left each tile is labeled with its virtual address. The page table shown in the middle maps that virtual address to a tile stored at an arbitrary offset in the physical backing storage, shown on the right.

![Tiles labelled by virtual addresses stored by offsets in the backing storage](images/demand_loading_4.png)

Of course, the level of indirection is not free, but it's hardware accelerated by the virtual memory system. The benefit is that the backing storage is quite compact, compared to the memory that would be required if physical memory was allocated for all the tiles in contiguous order.

## Library overview

The OptiX Demand Loading library provides a framework for loading CUDA sparse textures on demand. It is based on a multi-pass approach to rendering:

A page table tracks which tiles are resident on the GPU. As an OptiX kernel discovers missing tiles, it records page requests. After a batch of requests has been accumulated, the kernel exits to allow the page requests to be processed on the CPU. For each request, a tile is loaded from disk, decompressed, and copied to GPU memory. Once the required tiles have been loaded into GPU memory, the page table is updated and the OptiX kernel is relaunched. Any pixels with missing texture tiles are resampled. Each pass might encounter other missing tiles, for example due to dependent texture reads or newly followed ray paths, so the process is repeated until there are no more misses.

## Texture fetch

The OptiX Demand Loading library provides `tex2D`, `tex2DLod`, and `tex2DGrad` functions to fetch from demand-loaded sparse textures in the file [Texture2D.h](/DemandLoading/DemandLoading/include/OptiXToolkit/DemandLoading/Texture2D.h). These functions are implemented entirely in header code, and are templated on the return type.  For example, the signature for the `tex2DGrad` is as follows:

```
template <class Sample> 
Sample tex2DGrad( 
    const DeviceContext& context, 
    unsigned int textureId, 
    float x, float y, 
    float2 ddx, float2 ddy, 
    bool* isResident );
```

The arguments to `tex2DGrad` are:

- `Sample` -
The desired return type, such as `float4`.

- `context` -
The demand loading device context which contains the page table and other data.

- `textureId` -
The id of the demand-loaded texture that is to be sampled.

- `x, y` -
The texture coordinates (normalized from zero to one).

- `ddx, ddy` -
The gradients of the texture coordinates, which are used to determine the miplevel. The gradients are normally calculated from ray differentials or ray cones in an OptiX closest-hit shader.

- `isResident` -
An output parameter, which is set to true if the required tiles are resident in GPU memory.

When called, the tex2DGrad function performs the following steps, which are illustrated in the diagram below.

![Texture fetch steps ](images/demand_loading_5.png)

- First calculate the required miplevels based on the gradients. Two miplevels are typically selected, and the final result is a weighted average.

- Calculate the texture footprint (what texture tiles are required by the call), using the texture coordinates to determine the offsets of the texture samples within the required miplevel. Linear filtering employs multiple samples, so up to four tiles might be required per miplevel.

- For each required tile, calculate its index in the page table.

- If the required tiles are resident, execute the usual texture fetch instruction (bottom right of the diagram). This provides fully hardware accelerated sampling and filtering.

- For missing tiles, record a tile request (bottom left of the diagram). Typically the ray is then terminated by throwing an OptiX user exception. But this step is user-programmable, so alternatively a default color could be substituted to allow shading to continue. Of course that's an approximation that might only be acceptable in a preview render or as an interim step during progressive refinement.

## Page request processing

After a batch of tile requests has accumulated on the GPU, the OptiX kernel exits to allow them to be processed on the CPU.

As illustrated below, the tile requests are processed by a parallel loop that reads tiles from disk or some other image source, decompresses them as needed, and copies them into GPU memory. The OptiX Demand Loading library provides support for reading OpenEXR textures, block compressed textures (bc1-bc7) in .dds files, and other file formats through Open ImageIO; incorporating additional third-party libraries is straightforward.

When all the requests are processed, the OptiX kernel is relaunched for another rendering pass. Additional page requests might arise, for example due to dependent texture reads, so the entire process is repeated until there are no more requests. Most final frame renderers run multiple passes in any case, so the demand loading system can be seen as simply adding some startup passes to a render.

This fits quite naturally into an adaptive sampling framework. It's not necessary to keep track of which rays failed due to missing tiles. Those rays make no contribution to the framebuffer, so the adaptive sampler will naturally resample the missing pixels in the next kernel launch.

![An adaptive sampling framework](images/demand_loading_6.png)

## Host-side library

In addition to the texture fetch functions, which are implemented as CUDA header code, the OptiX Demand Loading library provides a C++ interface for host side operations. The primary interface is through the `DemandLoader`, which is obtained by calling the following function for the current CUDA device:

```
DemandLoader* createDemandLoader( const Options& options );
```

Note that a separate DemandLoader must be created for each device used.  This is in line with CUDA and OptiX programming, which require explicit CUDA contexts and launches for each device. Also, similar to CUDA host functions, DemandLoader method calls require that the current CUDA context correspond to the one for which the DemandLoader was created.

The following DemandLoader method creates a demand-loaded sparse texture. The texture descriptor specifies the filter mode and wrap mode, etc.

```
const DemandTexture& createTexture(
    std::shared_ptr<ImageReader> image,
    const TextureDescriptor& textureDesc);
```

When created, the texture initially has no backing storage, but texture tiles will be loaded when they are requested in an OptiX kernel. The `ImageReader` argument has a `readTile()` method that serves as a callback during page request processing.

The returned `DemandTexture` object is mostly opaque. It provides a `getId()` method that returns the texture identifier, which is required by the `tex2DGrad` function described earlier. In common use, the texture id would be associated with an object in the OptiX shader binding table (SBT), but it might also be passed to the GPU in a launch parameter.

Before launching an OptiX kernel, the host program calls `launchPrepare()` on the DemandLoader, which updates device-side data structures including the page table and an array of sparse texture samplers. The method returns a `DemandTextureContext` (by result parameter) that is required by the texture fetch functions. The host program typically passes the context to the OptiX kernel as a launch parameter after copying it to device memory.

```
void launchPrepare(
    unsigned int deviceIndex,
    DemandTextureContext& demandTextureContext);
```

After an OptiX kernel exits, the host program calls the `processRequests()` method on the DemandLoader, which processes any accumulated page requests. As mentioned, these are processed asynchronously in a parallel loop.  A `Ticket` object is returned from processRequests, and user code can query the ticket to track its progress, or wait on it to block the current thread until all of the requests have completed.

```
Ticket processRequests( CUstream stream, const DeviceContext& deviceContext )
```

Tip: In the usual case that many launches are performed in a loop, it is often more efficient to wait on the ticket just *before* calling `launchPrepare()`. This gives the main thread as much time as possible to do other work before blocking for the ticket to finish.

Once rendering has finished, the `destroyDemandLoader()` function can be used to destroy the DemandLoader, which frees all its resources, including device-side sparse texture memory.

```
void destroyDemandLoader(DemandLoader* manager);
```

The OTK [Texture](/examples/DemandLoading/Texture) sample shows basic use of the demand loading library for CUDA sparse textures.

# Advanced demand loader features

The OptiX demand-loaded texture library has matured into a high quality texturing system suitable for film quality renders.  It is used by several commercial renderers.  This section describes some of the advanced options and features of OptiX demand-loaded textures, and discusses some technical details, performance characteristics, and limitations of the system.

## Configuring the demand loader 

The OptiX Demand Loading Library can be configured using an [Options](/DemandLoading/DemandLoading/include/OptiXToolkit/DemandLoading/Options.h) struct passed to `createDemandLoader()`.  This gives the size of the virtual page table and other buffers used to transfer data, specifies memory limits for device-side texture data, and defines the number of threads and CUDA streams to use in the demand loader.  The Options struct and default values are shown below, and the next section describes the use of the options.

```
struct Options
{
    // Page table size
    unsigned int numPages            = 64 * 1024 * 1024;
    unsigned int numPageTableEntries = 1024 * 1024;

    // Demand loading
    unsigned int maxRequestedPages   = 8192;
    unsigned int maxFilledPages      = 8192;

    // Demand load textures
    unsigned int maxTextures         = 256 * 1024;
    bool useSparseTextures           = true;
    bool useSmallTextureOptimization = false;
    bool useCascadingTextureSizes    = false;
    bool coalesceWhiteBlackTiles     = false;
    bool coalesceDuplicateImages     = false;

    // Memory limits
    size_t maxTexMemPerDevice        = 0; // (0 = unlimited)
    size_t maxPinnedMemory           = 64 * 1024 * 1024;

    // Eviction
    unsigned int maxStalePages       = 8192
    unsigned int maxEvictablePages   = 0; // (not used)
    unsigned int maxInvalidatedPages = 8192; 
    unsigned int maxStagedPages      = 8192;
    unsigned int maxRequestQueueSize = 8192;
    bool useLruTable                 = true;
    bool evictionActive              = true;

    // Concurrency
    unsigned int maxThreads = 0; // (0 = hardware_concurrency)

    // Trace file
    std::string traceFile;
};
```

## Configuring the page table

The demand loading options `numPages`, `numPageTableEntries`, and `maxTextures` configure the page table that will hold residence information and table entries for texture samplers, base colors, texture tiles, texture cascades, and other demand-loaded assets.

`numPages` sets the size of the page table. All pages in the page table use 6 bits for **residence bits**, **request bits** and 4 bit **LRU counters**.

A slot in the page table will be reserved for each texture tile in the virtual texture set of an application, and in most cases these make up the bulk of the table. CUDA sparse texture tiles are 64k in size, so the maximum amount of addressable texture with the default configuration is roughly 4 TB, large enough for many production scenes. This can be extended using **cascading texture sizes**, described later.

Texture tiles do not need page table entries on the device, so to save space, only the first `numPageTableEntries` of the table have (8 byte) page table entries allocated for them. 

The `maxTextures` option defines the maximum number of CUDA textures that can be tracked by the demand loading system. Each texture has a page table entry for the sampler, and one for the base color, so `maxTextures` must be less than half `numPageTableEntries`.  Note also that the maximum number hardware textures that CUDA allows is about one million, so values above this don't make sense.

## Final frame vs. interactive rendering

By default, the Options struct is set up for final frame rendering (`maxRequestedPages` and associated fields (`maxFilledPages, maxStalePages, maxInvalidatedPages, maxStagedPages, maxRequestQueueSize`) are in the thousands). Individual applications may wish to tweak these based on their exact requirements. The options can be set for interactive rendering by reducing `maxRequestedPages` to a smaller value, such as 64 or 128. Currently, there is no option to guarantee a stable framerate, however.

## Other options

Other notable fields in the Options struct include:

- `useSparseTextures` -  When set to false, this flag configures the system to use standard (dense) textures instead of sparse textures. Whole textures are loaded on demand instead of individual texture tiles. This is faster than sparse textures for small texture sets, and can be useful for debugging.
    
- `useSmallTextureOptimization` - (currently not working becasue of CUDA limitations) Replaces small textures with dense textures to save memory. This is intended to handle the case of many small textures.
    
- `useCascadingTextureSizes` - Instantiates hardware sparse textures at a small initial size, and then expands them as needed to fill tile requests. Creating sparse textures in this way increases the virtual texture set that can be defined in the demand texturing system and reduces startup time for scenes with many textures.
    
- `coalesceWhiteBlackTiles` - This optimization combines black and white texture tiles for certain kinds of images, which saves memory in the common case of mask textures with large white or black regions.
    
- `coalesceDuplicateImages` - When turned on, this optimization combines identical images, using a hash of the mip tail to determine when textures are the same. Because it is hash-based, different files with identical images will still be coalesced.

- `maxTexMemPerDevice` - Set the maximum GPU memory to use for textures. If eviction is turned on, the demand loader will start eviction when this amount of texture is reached.
    
- `maxPinnedMemory` - The maximum amount of page-locked (pinned) memory to allocate for transfer buffers.

- `maxStagedPages` - Defines how many texture tiles will be set aside as unusable when eviction is active so that they can be used to fill tile requests from the next launch.

- `useLruTable` - Setting this option to false turns off the LRU table so that randomized eviction is used instead.
    
- `evictionActive` - Turn eviction on or off. Disabling eviction improves texturing speed when the texture working set will fit into GPU memory. 

- `maxThreads` - Sets the maximum number of host threads used to fill demand loading requests. Applications may wish to experiment with different sizes to determine the optimal value for their use case. Anecdotally, we have sometimes seen faster render times when `maxThreads` is set to 1 rather than maximum concurrency.

## Supported file formats

EXR images are supported using the [CoreEXRReader](/DemandLoading/ImageSource/include/OptiXToolkit/ImageSource/CoreEXRReader.h) class to wrap the EXR reading functions of the [OpenEXR](https://openexr.com/) library.  The older `EXRReader` class is deprecated as it does take advantage of the parallel processing capabilities available in OpenEXR 3.1.

Block compressed textures (BC1...BC7), stored as .dds files, are supported using the [DDSImageReader](/DemandLoading/ImageSource/include/OptiXToolkit/ImageSource/DDSImageReader.h) class. The block compressed formats provide substantial texture compression (2-8x) on the GPU while maintaining high image quality. The [NVIDIA Texture Tools](https://developer.nvidia.com/texture-tools-exporter) utility can convert other image types to .dds files. (Note that CUDA is limited to rendering mip levels for BC textures that are multiples of 4 in size. The DDSImageReader will truncate the mip pyramid as needed to maintain this requirement.)

Neural textures created by the [Neural Texture SDK](https://github.com/NVIDIA-RTX/RTXNTC). Neural textures can achieve extremely high compression ratios on the GPU for bundled texture sets (up to 50x or more). They take advantage of the GPU tensor cores to achieve fast decompression. 

The [OIIOReader](/DemandLoading/ImageSource/include/OptiXToolkit/ImageSource/OIIOReader.h) class wraps [OpenImageIO](https://sites.google.com/site/openimageio/home), allowing a wide range of image files to be read, including TIFF, JPEG, and PNG, as well as EXR. Note that for best performance, EXR files should be read via `CoreEXRReader`, however.

Other image file types can easily be added by defining a ImageSource subclass that reads them.

## Texture base colors

1x1 textures (and textures for which only the 1x1 mip level is accessed) are stored as half4 base colors directly in the page table. Since all base colors are stored in `half4` format, some precision loss may occur. Host side memory overhead for base colors is higher. ImageSource and TextureRequestHandler objects are stored on the host for all textures, so a different solution may be more practical if millions of base colors are needed.

## Small texture optimization

**The small texture optimization is not currently working because of a CUDA limitation that prevents sparse and dense textures from being used at the same time. Using all dense textures does work, however.**

Because sparse texture tiles are always 64 KB in size, small sparse textures can waste GPU memory.  To save space, the OptiX Demand Loading library can instantiate very small textures (at most 1024 pixels on mip level 0) as dense (standard) textures. When used, dense textures are loaded on demand when first referenced. Once loaded they are not evicted.  It is practical to create and use up to a million small dense textures, assuming sufficient memory on the device.  For example, a million mipmapped 16x16 half4 textures would use about 2.6 GB of GPU memory.

The OptiX Demand Loading Library reverts to dense textures regardless of size on older hardware that does not support hardware sparse textures. The Options struct also allows the application to specify all dense textures if desired.

## Host and device filled textures

Depending on the ImageSource, texture requests can be filled on the host and then transferred to the device, or filled directly on the device by a CUDA kernel. The file [DeviceMandelbrotImage.h](/DemandLoading/ImageSource/include/OptiXToolkit/ImageSource/DeviceMandelbrotImage.h) gives an example of device-side fulfillment.

## Preloading, unloading, and replacing textures

Most of the actions to load and unload textures in the demand loading library are automatic. However, the `DemandLoader` includes some functions to load and unload textures and texture tiles without waiting for texture requests. These include:

- `initTexture` - Initialize the texture.
- `loadTextureTiles` - Load all of the texture tiles for a texture.
- `loadTextureTile` - Load or replace a specific texture tile.
- `unloadTextureTiles` - Discard all texture tiles for a texture on the next `pullRequests()` call.
- `invalidatePage` - Discard the page (of a texture tile or other resource) on the next `pullRequests()`.
- `replaceTexture` - Replace the image source for a texture, discarding any resident tiles.

The [texture painting](/examples/DemandLoading/TexturePainting) sample shows how to use many of these.

## Setting the max texture tile memory

The `maxTexMemPerDevice` field of the Options struct determines the initial amount of device memory that will be used before eviction starts. This value can be changed after creating the demand loader by calling `setMaxTextureMemory`.  If the new size is less than the amount currently allocated, the demand loader deletes some of the texture memory arenas and discards any tiles stored in them (they can be reloaded if requested again). In this way, an application can shrink or grow the amount of texture memory that it dedicates to texturing based on changing needs.

## Cascading texture sizes

By default, the demand loading library instantiates CUDA textures at the size of the image source, which can be inefficient if the finer mip levels of the texture are never accessed. When `useCascadingTextureSizes` is set to true, the demand loader instead creates CUDA textures that are just large enough to fill requested mip levels.  If finer mip levels are needed on a later launch, the texture is *cascaded* (reinstantiated) to a larger size. 

Texture cascading offers two main benefits. First, it expands the virtual texture set size that can be managed by the demand loader, sometimes by an order of magnitude or more. Second, it reduces startup times for textures by a similar amount, since sparse texture creation time is dependent on texture size in CUDA. As an example, I ran the [udimTextureViewer](/DemandLoading/UdimTextureViewer) with the argument `--udim=50x50` to create 2500 8K textures. This took 2.3 seconds on a 5080 with texture cascading, but 30 seconds without it.

## Eviction 

The OptiX Demand Loading library supports eviction of texture tiles. Enabling eviction in user code is a matter of setting appropriate values in the Options struct, including `maxTexMemPerDevice`, and setting `evictionActive` to true. The DemandLoader takes care of the details.

Eviction operates by repurposing pages that are not in use. A page that is resident on the device, but was not used in the most recent launch is said to be "stale".  The `processRequests()` method in the DemandLoader gathers a list of stale pages from the device, returning least recently used pages if the LRU table is active, or random stale pages otherwise.  Pages can be flagged as non-evictable by setting the LRU value for the page to NON_EVICTABLE_LRU_VAL when they are loaded. This is done for samplers and base colors.

When eviction is active, some of the stale pages are "staged", meaning that they are flagged as non-resident so that they can be reclaimed immediately when needed.  The field `maxStagedPages` in the Options struct gives the size of the staged pages list that the demand loading system will try to maintain.  We recommend setting maxStagedPages to be about 5-10 percent of the total GPU memory set aside for textures.
For example, if maxTexMemPerDevice is set to 2048 (2 GB), OptiX demand loading will allocate about 32K texture tiles in total, and a reasonable value for maxStagedPages is 3200.

If an OptiX program requests a staged page, the page is considered to be non-resident on the device, even though the resource is still there. Requests for staged pages are filled by simply setting the value in the page table back to "resident" on the next processRequests cycle. The data does not have to be reloaded (called the second chance algorithm).

Maintaining the state of the page table and LRU counters when eviction is active carries some overhead. Texture ops take about 3 times longer when eviction is active compared to sampling a resident texture when eviction is not active (although actual performance reduction will likely be much smaller, even for texture heavy launches). The DemandLoader provides the function `enableEviction()` to turn eviction on or off on a per launch basis for performance.

## UDIM textures

Many modeling and rendering packages support UDIM textures, which map a grid of textures to a single UV space.  UDIMs allow a texture to be split into multiple files that are authored separately, which gets around size limits for individual images, and permits different parts of a texture to be authored at different resolutions.

The `DemandLoader` class provides convenience methods to define and sample UDIM textures. The following method is provided in the DemandLoader class to define UDIM textures:

```
const DemandTexture& createUdimTexture( 
    std::vector<std::shared_ptr<imageSource::ImageSource>>& imageSources,
    std::vector<TextureDescriptor>& textureDescs,
    unsigned int udim,
    unsigned int vdim,
    int          baseTextureId,
    unsigned int numChannelTextures = 1);
```

`createUdimTexture` returns a DemandTexture reference, and its id should be passed to the texture sampling functions in device code. The parameters are as follows:

- `imageSources` - Image readers for the textures in the subtexture grid, listed in row-major order. A nullptr can be used to indicate a hole in the image sequence. If there are not enough images to fill the grid, the remaining images are considered to be null as well.

- `textureDescs` - Texture descriptions for each of the subtextures, giving addressing and interpolation modes.

- `udim, vdim` - Dimensions of the UDIM grid. In typical UDIM systems, udim is set to 10, and vdim is allowed to float. For the demand loading library, `udim` can vary but must correspond to how the subtextures are layed out in the `imageSources` list, and `vdim` must be large enough to fit all the textures in the list.

- `baseTextureId` - A "base texture" is an optional single texture that covers the whole UDIM grid. `baseTextureId` tells which previously defined texture to use for the base texture. A value of -1 indicates that no base texture is present. The point of a base texture is that far away objects will not have their subtextures instantiated, reducing GPU texturing overhead. 

- `numChannelTextures` - When set to a value other than 1, it allows multiple channel textures to be defined for each subtexture at the same time, for example diffuse color, specular color, roughness, or metalness.

To sample a UDIM texture, use the device side method `tex2DGradUdim()` defined in `Texture2DExtended.h`, which has the same semantics as `tex2DGrad()`, but handles switching between different subtextures in the UDIM grid: 

```
template <class TYPE> TYPE tex2DGradUdim( 
    const DeviceContext& context, 
    unsigned int textureId, 
    float x, 
    float y, 
    float2 ddx, 
    float2 ddy, 
    bool* isResident );
```

The `tex2DGradUdim()` method can also be used for non-UDIM textures with almost no performance penalty, so we suggest always using it in applications that support UDIM textures, rather than switching between texture sampling functions in a closest hit program. To minimize artifacts at texture boundaries, `CU_TR_ADDRESS_MODE_CLAMP` address mode should be used for subtextures defined for use with tex2DGradUdim.

The function `tex2DGradUdimBlend()` interpolates between adjacent textures in the UDIM grid to create a seamless transition, whereas `tex2DGradUdim()` does not. The interpolation incurs a slight performance penalty, since multiple texture samples must be sampled on subtexture edges. When sampling with tex2DGradUdimBlend, `CU_TR_ADDRESS_MODE_BORDER` address mode should be used so that the textures blend properly across subtexture boundaries.

## Managing large texture working sets

The set of texture tiles that must be resident during a kernel launch can be termed the texture **working set** for the launch.  In typical usage, rendered pixels cover roughly 4 texels per texture layer, or about 32 bytes per texture layer per pixel for `half4` textures. If a scene's working set is much larger than indicated by the number of texture layers, the renderer may be sampling textures at too high resolution. Common causes of oversampling include

- Using a distance-based LOD metric (or no metric) instead of proper ray cones or differentials.
- Having too narrow a starting angle for ray cones or differentials.
- Using a single ray cone instead of two.
- Loading non-tiled or non-mipmapped textures.
- Not enabling eviction for multi-launch renders.
- Setting `useSparseTextures` flag to false in the demand texture options struct.

**Reducing the working set.**  Some scenes simply have very large working sets. Applications can reduce the working set by employing some combination of different methods, including:

- Texture coalescing

    * The demand loading options `coalesceWhiteBlackTiles` and `coalesceDuplicateImages` direct the texturing system to coalesce duplicate textures and white/black texture tiles. Enabling these options will reduce the working set for some scenes.

- Tiled rendering

    * *Tiled rendering* refers to subdividing an image into a grid of tiles that are rendered in separate launches. Tiled rendering can drastically reduce the *first hit* working set per launch. The *secondary hit* working set will also be reduced, although not to the same degree.

- Compressed textures

    * Film renders typically use uncompressed textures to achieve the highest visual quality, relying on texture caching to manage the texture working set. GPU memory constraints are tighter than CPU constraints, however. To address this, an application may choose to render with compressed textures, either for all ray hits, or for secondary hits. The OTK demand texturing system supports **Block Compressed (BC) textures** and **Neural Textures (NTC)**. The [compressedTextureCache](/examples/DemandLoading/CompressedTextureCache/) utility can convert images to BC formats, and the [Neural Texture SDK](https://github.com/NVIDIA-RTX/RTXNTC) can convert to ntc format.
    
- Mip level bias

    * A *mip level bias* changes the mip level sampled by a texture call, reducing the working set at the expense of slight blurring. The bias can be added either by setting the `mipmapLevelBias` field of the `CUDA_TEXTURE_DESC` struct when defining the texture, or scaling the gradients sent to the `tex2DGrad*` call. An application could bias all texture hits, or just secondary hits.  The exact amount of savings is view dependent, but a rough rule of thumb is that a half mip level bias will reduce the working set by 50%.

## Cubic filtering and texture derivatives

High quality renderers often employ cubic filtering when magnifying a texture to reduce blocky image artifacts.  [Open ImageIO](https://github.com/AcademySoftwareFoundation/OpenImageIO) is an industry standard texturing library used by many commercial renderers. It supports four filtering modes: *point*, *linear*, *bicubic*, and *smart bicubic*, and computes texture derivatives in the chosen mode. The OTK demand loading library includes sampling functions that attempt to work like the OIIO `texture()` function.  It supports the same four filtering modes as Open ImageIO, and can calculate texture derivatives in addition to filtered texture values. These Open ImageIO work-alike functions are header only implementations, and are found in the file [Texure2DCubic.h](/DemandLoading/DemandLoading/include/OptiXToolkit/DemandLoading/Texture2DCubic.h) with the main entry points being `textureCubic`, `textureUdim`, and `texture`.  The `textureCubic` function looks like this:

```
template <class TYPE> bool textureCubic( 
    const DeviceContext& context, 
    unsigned int textureId, 
    float s, 
    float t, 
    float2 ddx, 
    float2 ddy, 
    TYPE* result, 
    TYPE* dresultds, 
    TYPE* dresultdt );
```

The filtered texture result is returned in three return parameters,`result`, `dresultds`, and `dresultdt`. 

To perform cubic filtering, the `filterMode` field in the `TextureDescriptor` struct must be set to `FILTER_SMARTBICUBIC` or `FILTER_BICUBIC` when defining the texture, and one of the texturing functions in `Texture2DCubic.h` must be used to sample the texture.

The main cubic filtering routines are defined in the file [CubicFiltering.h](/ShaderUtil/CubicFiltering.h), which is in ShaderUtil, outside of the demand loading library, so they can be used without it.

## Stochastic texture filtering

[Stochastic texture filtering](https://research.nvidia.com/publication/2024-05_filtering-after-shading-stochastic-texture-filtering) achieves high quality texture filtering by jittering texture coordinates and combining samples rather than computing weighted sums of many texture reads. All of the texture sampling functions in the OptiX demand loading library have variants to support stochastic filtering. The augmented entry points include an extra `float2` parameter for jittering the texture coordinate at the sampled mip level.The OTK stochastic texturing sample demonstrates how to do stochastic texture filtering with the OptiX Toolkit. The OTK sample [stochasticTextureFiltering](/DemandLoading/StochasticTextureFiltering) gives an examples of this.

Similar to cubic filtering, the stochastic texturing functions are defined in ShaderUtil in the file [stochastic_filtering.h](/ShaderUtil/stochastic_filtering.h), so they do not depend on the demand loading.

## Ray cones for texture streaming

Renderers employing texture streaming often rely on **ray differentials** or **ray cones** to drive texture requests. ShaderUtil provides an implementation of ray cones in the file [ray_cone.h](/ShaderUtil/ray_cone.h). The OTK [rayCones](/examples/DemandLoading/RayCones) sample shows how to use ray cones in a ray tracer.

## Limits on demand loaded textures

Demand loaded textures have the same size limits as CUDA textures.  The maximum dimensions for a texture are dependent on the pixel format and whether mipmapping is used. Some size limits for mipmapped textures in OptiX using SM 2.x are as follows:

| Format | Max resolution |
| ------- | -------------- |
| `float4` | 16384 x 8192 (or 8192 x 16384) |
| `float2, half4` | 16384 x 16384 |
| `float, half2, ubyte4, uint` | 32768 x 16384 |
| `half, ubyte2, ubyte` | 32768 x 32768 |

The number of individual textures (sparse or otherwise) that can be defined in OptiX demand loading is defined by the `maxTextures` field in the Options struct, and by default is 256K textures. In theory up to a million sparse textures can be defined in CUDA, but the number of active sparse textures (textures with at least one tile resident on the GPU) is limited by the hardware texture tile size (64 KB), which boils down to 16K texture tiles per GB of texture data reserved on the device.  

The total number of virtual tiles that can be managed by OptiX Demand Loading is limited by the `numPages` field in the Options struct (about 4TB in the default configuration). The  virtual texture set size for a scene can be much larger when cascading texture sizes is turned on, however, perhaps 100 TB for some scenes.

## Eviction limits

Texture tiles can be evicted and reused, but tile pool backing storage is not freed until the DemandLoader is destroyed unless the application explicitly calls `setMaxTextureMemory`. Once created, CUDA sparse texture objects are not evicted, and the associated page table entries are not recycled. Dense textures are also not evicted.

The system does not stage texture tiles unless they are stale, and only staged tiles can be used to fill requests once the initial tile pools have been exhausted.  Consequently, if the working set needed for a launch is too large, the launch will always have non-resident texture references.  Bucketed rendering (rendering different parts of an image in separate launches) can reduce the working set.

## Device-side overheads

The demand loading library allocates a number of tables on the device to manage demand loaded resources. With existing defaults, the paging system uses about 64 MB of device memory. Also, each texture larger than 1x1 that is instantiated takes 128 bytes for a sampler object on the device. 

CUDA maintains hardware tables with 8 bytes per virtual tile for instantiated sparse textures. To put this in perspective, if sparse textures spanning all of the default page table were instantiated, the overhead would be 512 MB.

## Host-side overheads

The host maintains a sparse page table per device, storing resident page entries in a set of maps.

A pinned memory buffer is used to transfer texture data, with a maximum size given in the Options struct (64 MB is the default per device).

Each texture defined in the system has an ImageReader object and a TextureRequestHandler.

