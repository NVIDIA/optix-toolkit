# OptiX Toolkit changes

## Version 1.1.x
* Improved CMake installation logic.
* Added NeuralTextures library for neural texture compression support (see the [README](NeuralTextures/README.md)).

## Version 1.0.1
* Added support for block compressed formats (.dds files) in the demand load library.
* Made numerous improvements to cubic filtering.
* Fixed a texture tile indexing error in DemandLoading
* Fixed pixel stride calculation in PbrtAlphaMapImageSource
* Enhanced compatibility: added support for CUDA 13.0, OptiX 8.1+, and fixed various build issues (Windows, OptiX, C++ standards).
* Added support for installation of OptiX Toolkit.
* Refactored and improved debug utilities and statistics reporting.
* Improved test coverage for demand loading and geometry.
* Refactored code for better maintainability: extracted and renamed libraries, improved modularity, and cleaned up includes.
* Updated dependencies and build scripts for better compatibility with vcpkg and OpenImageIO 3.x.
* Improved logging, error handling, and documentation throughout the toolkit.
* Various bug fixes, code cleanups and enhancements.

## Version 1.0.0
* Merged multiple submodules into the main repository (CMake, DemandLoading, Memory, OmmBaking, PyOptiX, ShaderUtil, examples).

## Version 0.9.4
* Renamed `DemandLoader::unloadResource()` to `invalidatePage()`

## Version 0.9.3
* The `DemandLoader` interface now provides a `setPageTableEntry` method, which is helpful for
  asynchronous resource request handling.  In such an approach `ResourceCallback` can enqueue a
  request and return false, indicating that the request has not yet been satisfied.  Later, when the
  request has been processed, `setPageTableEntry` can called to update the page table.  Note that
  it's not necessary to call this method when requests are processed synchronously: if the
  `ResourceCallback` returns true, the page table is automatically updated.
* embed_cuda now generates OptiX IR by default.  Be careful to use the PTX option when compiling pure CUDA kernels.

## Version 0.9.2
* Added CMake presets.  Updated [README](README.md).
* Overhauled vcpkg dependency management and build options.
* Fixed intermittent PTX compilation issues caused by lack of trailing zero bytes in embedded PTX.

## Version 0.8

* Added [Self Intersection Avoidance library](ShaderUtil/docs/selfIntersectionAvoidance/README.md).
* Added [DemandGeometry library](DemandLoading/DemandGeometry/README.md)
* Added [DemandGeometryViewer example](examples/DemandLoading/DemandGeometryViewer)
* OTK now requires cmake 3.24 or later, leveraging a [new feature of FetchContent](https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html#fetchcontent-and-find-package-integration) that invokes find_package and uses installed third-party packages when possible. 
* Added a new [Memory](Memory/README.md) submodule that provides device memory allocators, etc.
  * The demand loading library now depends on this submodule.
* The Demand Loading library now supports a wide range of image file types (via OpenImageIO).  See 
the [Demand Loading CHANGELOG](DemandLoading/CHANGELOG.md).

## Version 0.6 - 2022-01-18

* Added OmmBaking library in commit b4810b0.
* Added a simple OptiX example (OtkHello) in commit 7db1ace.
* Added demandTextureViewer app in commit b3c418c.
* Added DemandLoading overview from OptiX Programming Guide in commit 00dce9b.
* Partial switch from CUDA runtime to drive API in commit e9898e7.
