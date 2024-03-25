# OptiX Toolkit changes

More detailed change logs are available in various submodules, e.g. 
[Demand Loading CHANGELOG](https://github.com/NVIDIA/otk-demand-loading/blob/master/CHANGELOG.md).

## Version 0.9.2
* Added CMake presets.  Updated [README](README.md).
* Overhauled vcpkg dependency management and build options.
* Fixed intermittent PTX compilation issues caused by lack of trailing zero bytes in embedded PTX.

## Version 0.8

* Added [Self Intersection Avoidance library](https://github.com/NVIDIA/otk-shader-util).
* Added [DemandGeometry library](https://github.com/NVIDIA/otk-demand-loading/tree/master/DemandGeometry)
* Added [DemandGeometryViewer example](https://github.com/NVIDIA/otk-examples/tree/master/DemandLoading/DemandGeometryViewer)
* OTK now requires cmake 3.24 or later, leveraging a [new feature of FetchContent](https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html#fetchcontent-and-find-package-integration) that invokes find_package and uses installed third-party packages when possible. 
* Added a new [Memory](https://github.com/NVIDIA/otk-memory) submodule that provides device memory allocators, etc.
  * The demand loading library now depends on this submodule.
* The Demand Loading library now supports a wide range of image file types (via OpenImageIO).  See 
the [Demand Loading CHANGELOG](https://github.com/NVIDIA/otk-demand-loading/blob/master/CHANGELOG.md).

## Version 0.6 - 2022-01-18

* Added OmmBaking library in [b4810b0](commit/b4810b0)
* Added a simple OptiX example (OtkHello) in [7db1ace](commit/94628f28f05e6b19b4c956b53d06bf6d37db1ace)
* Added demandTextureViewer app in [b3c418c](commit/c8643dc18726ba7ae12a3821884b97901b3c418c)
* Added DemandLoading overview from OptiX Programming Guide in [00dce9b](commit/d139700afa7b3841c9d1b8938d4eca72e00dce9b)
* Partial switch from CUDA runtime to drive API in [e9898e7](commit/92a30c3b195286b30f3186662b175f968e9898e7)
