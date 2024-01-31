# OptiX Demand Loading Library Change Log

## Version 0.9

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
