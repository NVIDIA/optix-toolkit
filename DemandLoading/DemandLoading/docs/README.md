
# Demand-loaded sparse textures

This section describes the OptiX Demand Loading library, which is a standalone CUDA library that is distributed as open-source code in the OptiX SDK.
(It is free for commercial use.)
Although it is part of the OptiX SDK, the Demand Loading library has no OptiX dependencies, and it is technically not part of the OptiX API.

The OptiX Demand Loading library allows hardware-accelerated sparse textures to be loaded on demand, which greatly reduces memory requirements, bandwidth, and disk I/O compared to preloading textures.
It works by maintaining a page table that tracks which texture tiles have been loaded into GPU memory.
An OptiX closest-hit program fetches from a texture by calling library device code that checks the page table to see if the required tiles are present.
If not, the library records a page request, which is processed by library host code after the kernel has terminated.
Kernel launches are repeated until no tiles are missing, typically using adaptive sampling to avoid redundant work.
Although it is currently focused on texturing, much of the library is generic and can be adapted to load arbitrary data on demand, such as per-vertex data like colors or normals.

Before describing the design and implementation of the library in more detail, we first provide some motivation and background information.

## Motivation

The size of film-quality texture assets has been a longstanding obstacle to rendering final-frame images for visual effects shots and animated films.
It's not unusual for a hero character to have 50-100 GB of textures, and many sets and props are equally detailed (sometimes for no good reason, except that it would be more expensive to author at a more appropriate resolution).

Film models are usually painted at extremely high resolution in case of extreme closeup.
A character's face is typically divided into several regions (for example, eye, nose, mouth), each of which is painted at 4K or 8K resolution.
In addition, numerous layers are often used, including coarse and fine displacement and various material layers such as dirt, each of which is a separate collection of textures.

It's not practical to preload entire film-quality textures into GPU memory.
Besides wasting GPU memory, it also requires time and bandwidth to read the textures from disk and transfer them to GPU memory.
Nor is it possible to downscale the images, since the level of detail that is required depends on the distance of the textured object from the camera.
Simply clamping the maximum resolution can lead to a loss of detail that is not acceptable in a film-quality renderer.

To make matters worse, it's not unusual for a production shot to have thousands of textured objects that are outside the camera view or completely occluded by other objects.
It's generally not possible to know *a priori* which textures are required without actually rendering a scene.

All of these factors make it difficult to load textures into GPU memory before rendering.
Our approach is a multi-pass one: the initial passes identify missing texture data, which is loaded from disk on demand and transferred to GPU memory between passes.

## Background

Mipmapping is a well known technique for dealing with high resolution textures.
The idea is to precompute downscaled images, as illustrated below, that can be used when a lower level of detail is acceptable, such as when a textured object is far from the camera.

![Mipmapped texture showing downscaled images](images/demand_loading_1.png)

In a ray tracer, choosing which level of detail to use is usually accomplished using ray differentials to determine the gradients of the texture coordinates.
The texture hardware on the GPU then uses those gradients to choose two miplevels, and then it performs a filtered lookup in each miplevel, and blends between the two.

Needless to say, that's a very expensive operation.
But it's necessary to avoid temporal aliasing when the camera or the geometry is moving, which would otherwise create a pop when the choice of miplevels suddenly changes from one frame to the next.
GPU acceleration for mipmapped texturing is often taken for granted, but it offers a huge performance gain compared to texturing on the CPU.

### Sparse textures

Mipmapped textures are usually tiled, which divides each miplevel into a number of smaller sub-images.
This allows a sparse memory layout to be used.
Instead of loading an entire miplevel onto the GPU, only the individual tiles that are required are loaded, which saves memory and bandwidth.

For example, consider rendering a quad angled away from the camera, as shown below.
The region that is closest to the camera must use tiles from a high resolution miplevel to ensure adequate detail.
However, tiles from coarser miplevels can be used farther from the camera.

![Texture-mapped quad angled away from the camera](images/demand_loading_2.png)

For example, the tiles highlighted below illustrate which tiles might be required:

![Tiles require](images/demand_loading_3.png)

Sparse textures are now supported in CUDA, starting with the CUDA 11.1 toolkit.
Under the hood, CUDA sparse textures employ the virtual memory system in a clever way.
The main complication is that texture filtering often requires samples from multiple tiles for a single texture fetch.
In that case, the GPU texture units require those tiles to be contiguous in memory.
But it would be wasteful to reserve enough space to allow that.

The trick is to use the virtual memory system to provide the illusion of contiguous tiles to the GPU texture units.
That's accomplished by binding the texture sampler to virtual memory, and then mapping individual tiles as virtual pages.

The diagram below illustrates how that works.
On the left each tile is labeled with its virtual address.
The page table shown in the middle maps that virtual address to a tile stored at an arbitrary offset in the physical backing storage, shown on the right.

![Tiles labelled by virtual addresses stored by offsets in the backing storage](images/demand_loading_4.png)

Of course, the level of indirection is not free, but it's hardware accelerated by the virtual memory system.
The benefit is that the backing storage is quite compact, compared to the memory that would be required if physical memory was allocated for all the tiles in contiguous order.


## Library overview

The OptiX Demand Loading library provides a framework for loading CUDA sparse textures on demand.
It is based on a multi-pass approach to rendering:

A page table tracks which tiles are resident on the GPU.
As an OptiX kernel discovers missing tiles, it records page requests.
After a batch of requests has been accumulated, the kernel exits to allow the page requests to be processed on the CPU.
For each request, a tile is loaded from disk, decompressed, and copied to GPU memory.
Once the required tiles have been loaded into GPU memory, the page table is updated and the OptiX kernel is relaunched Any pixels with missing texture tiles are resampled.
Each pass might encounter other missing tiles, for example due to dependent texture reads or newly followed ray paths, so the process is repeated until there are no more misses.

## Texture fetch

The OptiX Demand Loading library provides the following CUDA function (implemented entirely as header code), which used to fetch from a demand-loaded sparse texture with four floating-point channels.
(Additional overload functions are available for other formats.
Only 2D textures are currently supported.)

```
float4 tex2DGrad(
    const DemandTextureContext& context,
    unsigned int textureId,
    float x, float y,
    float2 ddx, float2 ddy,
    bool* isResident);
```

The arguments are as follows:

- `context`
Contains the page table and other data.

- `textureId`
Used to determine the offset in the page table at which the texture tiles are tracked.

- `x`
The texture coordinates (normalized from zero to one).

- `ddx`
The gradients of the texture coordinates, which are used to determine the level of detail required (that is, which miplevel).
The gradients are usually calculated from ray differentials in an OptiX closest-hit shader.

- `isResident`
An output parameter, which is set to true if the required tiles are resident in GPU memory.

The following steps are performed, as illustrated below.

![Texture fetch steps ](images/demand_loading_5.png)

First the required miplevels are calculated, which is based on the gradients.
As usual, two miplevels are typically selected, and the final result is a weighted average.

- The texture footprint is then calculated, using the texture coordinates to determine the offsets of the texture samples within the required miplevel.
Linear filtering employs multiple samples, so up to four tiles might be required per miplevel.

- For each required tile, its page table index is calculated by adding the tile index to the offset of this texture's page table entries.

- If the required tiles are resident, the usual texture fetch instruction is executed, as shown on the right side of the illustration.
That provides fully hardware accelerated sampling and filtering.

- If a tile is missing a tile request is recorded (as shown on the left).
Typically the ray is then terminated by throwing an OptiX user exception.
But this step is user-programmable, in the closest hit program.
So alternatively a default color could be substituted to allow shading to continue.
Of course that's an approximation that might only be acceptable in a preview render or as an interim step during progressive refinement.

## Page request processing

After a batch of tile requests has accumulated on the GPU, the OptiX kernel exits to allow them to be processed on the CPU.

As illustrated below, the tile requests are processed by a parallel for loop that reads tiles from disk, decompresses them, and copies them into GPU memory.
The OptiX Demand Loading library provides support for reading OpenEXR textures; incorporating other third-party libraries is straightforward.

When all the requests are processed, the OptiX kernel is relaunched for another rendering pass.
Additional page requests might arise, for example due to dependent texture reads, so the entire process is repeated until there are no more requests.

This fits quite naturally into an adaptive sampling framework.
It's not necessary to keep track of which rays failed due to missing tiles.
Those rays make no contribution to the framebuffer, so the adaptive sampler will naturally resample the missing pixels in the next kernel launch.

![An adaptive sampling framework](images/demand_loading_6.png)

## Host-side library

In addition to the texture fetch function described above, which is implemented entirely as CUDA header code, the OptiX Demand Loading library provides a C++ interface for host side operations.

The primary interface is the `DemandTextureManager`, which is obtained from the following function by providing a list of active devices and a few configuration options:

```
DemandTextureManager*
    createDemandTextureManager(
        const std::vector<unsigned int>& devices,
        const DemandTextureManagerConfig& config);
```


The following method creates a demand-loaded sparse texture.
The texture descriptor specifies the filter mode and wrap mode, etc.

```
const DemandTexture& createTexture(
    std::shared_ptr<ImageReader> image,
    const TextureDescriptor& textureDesc);
```

The texture initially has no backing storage.
The `ImageReader` argument has a `readTile()` method that serves as a callback during page request processing.

The returned `DemandTexture` object is mostly opaque.
It provides a `getId()` method that provides the texture identifier, which is required by the texture fetch function described above.
The texture id would typically be associated with an object in the OptiX shader binding table (SBT).

Before launching an OptiX kernel, the host program calls the following `DemandTextureManager` method, which updates device-side data structures including the page table and an array of sparse texture samplers.
The method returns a context (by result parameter) that is required by the texture fetch function.
The host program typically passes the context to the OptiX kernel as a launch parameter after copying it to device memory.

```
void launchPrepare(
    unsigned int deviceIndex,
    DemandTextureContext& demandTextureContext);
```

After an OptiX kernel has exited, the host program calls the following `DemandTextureManager` method, which processes any accumulated page requests.
As described above, the tile requests are processed by a parallel for loop that reads tiles from disk, decompresses them, and copies them into GPU memory.

```
int processRequests();
```

When rendering is complete, the following function is used to destroy the `DemandTextureManager`, which frees all its resources, including device-side sparse texture memory.

```
void destroyDemandTextureManager(DemandTextureManager* manager);
```

## Implementation

The full source for the OptiX Demand Loading library is provided in the OptiX SDK, along with a sample that illustrates its use.
It is free for commercial use.
The source code is accompanied by documentation (generated by Doxygen) that describes the implementation in detail.
Questions are welcome on the NVIDIA OptiX forums.

We expect that some developers will fork the library to customize it, while others will rely on NVIDIA for continued feature development.
The source code might move to a public repository in the future to facilitate such uses, with periodic library deliveries as part of the OptiX SDK.
