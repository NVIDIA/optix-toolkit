# OptiX Demand Loading Library Change Log

## Version 0.9

* The `DemandLoader` API has changed to simplify the handling of multiple GPUs and the management of the corresponding CUDA contexts.

    * Previously a single `DemandLoader` instance managed multipe GPUs, and several methods took a
      `deviceIndex` parameter specifying which GPU should be operated on.
    * Now the client application should create a CUDA context and call `createDemandLoader()` for each GPU. 
    * When calling `DemandLoader` methods, the client application must ensure that the current CUDA context matches the 
      context that was active when that `DemandLoader` was created.

* The demand texturing library now supports cascading texture sizes.  This feature can be turned on
  by setting the `useCascadingTextureSizes` field in the demand loading `Options` struct.  When
  enabled, cascading texture sizes instantiates hardware sparse textures at a small initial size,
  and then expands them as needed to fill tile requests. Creating sparse textures in this way has
  the benefits that:

    * It increases the virtual texture that can be defined in the demand texturing system, from 4 TB to
      over 100 TB in some scenes.
    * It reduces startup time for scenes with many tetxures since small sparse textures take less time
      to instantiate than large ones.

* The error message was improved when loading a texture sampler fails during request processing.
* A `Tile` structure was introduced to clearly distinguish tile coordinates vs. pixel coordinates.
* A `MipMapImageSource` adapater was added that adapts a single level image source into a mipmapped
  image source, using point sampling to generate smaller mip levels.
* A `TiledImageSource` adapater was added that adapts a non-tiled image source into a tiled image
  source by fetching mip levels into a memory buffer and serving tiles from there.
* The `MockOptix` google mock class was expanded to cover the entire OptiX API.
* PNG and JPEG image source tests were enabled when OpenImageIO is available.
* The Demand Geometry library received some bug fixes related to updating internal data structures.
* The Demand Material library was extracted from the Demand Geometry Viewer sample.

## Version 0.8

The demand loading library was factored into two layers:
- A layer that deals only in requested page ids
- A layer that maps requested page ids to textures and resources

This change should not affect clients of the existing demand loading
library.  Applications that wish to employ their own mapping of requested
pages to resources can use the lower layer.

A simple demand load geometry library was added.  Instead of loading geometry
directly into the scene, the application adds proxies associated with a bounding
box.  After launch, the demand geometry library reports back which proxies were
intersected and the application can use this to load the corresponding geometry.
See the [DemandGeometry README](DemandGeometry/README.md) for details on the API.

The demand loading library now depends on the [otk-memory](https://github.com/NVIDIA/otk-memory)
repository, which is a sibling submodule in the [optix-toolkit](https://github.com/NVIDIA/optix-toolkit) 
repository.

The ImageSource library now provides an `OIIOReader` class that wraps 
[OpenImageIO](https://sites.google.com/site/openimageio/home).
This allows a wide range of image files to be read, including TIFF, JPEG, and PNG.
* For best performance, EXR files should be read via `CoreEXRReader`, not `OIIOReader`.
* Use `ImageSource::createImageSource` to create an appropriate ImageSource based on a filename extension.
* The `OIIOReader` class is built only when OpenImageIO is found during CMake configuration.
  * The [vcpkg](https://vcpkg.io/en/getting-started.html) package system is a convenient way
    to install OpenImageIO and its prerequisites.
