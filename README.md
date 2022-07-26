
# OptiX Toolkit

A set of utilities commonly used in applications utilizing the [OptiX ray tracing API](https://developer.nvidia.com/rtx/ray-tracing/optix).

## Current Utilities
- **[PyOptiX](PyOptiX/README.md)** - Complete Python bindings for the OptiX host API.
- **[DemandLoading](DemandLoading/README.md)** -  a C++/CUDA library for loading CUDA sparse textures on demand in OptiX renderers.
- **[ImageSource](ImageSource/README.md)** - wrapper for OpenEXR image library (adaptable to other image formats).
- **[Gui](Gui/README.md)** - convenience code for incorporating OpenGL into OptiX applications.
- **[Cuda](Cuda/README.md)** - vector math and other CUDA helper functions for OptiX kernels.
- **[Util](Util/README.md)** - file handling and other utility functions.

## Requirements

- OptiX 7.4 or later.
- CUDA 11.1 or later.
- C++ compiler (e.g. gcc under Linux, Visual Studio under Windows)
- CMake 3.23 or later.  Using the latest CMake is highly recommended, to ensure up-to-date CUDA
language support.
- git (any modern version).

## Building OTK

- In the directory containing the OTK source code, create a subdirectory called `build` and `cd` to that directory.
```
mkdir build
cd build
```
- Configure CMake, specifying the location of the OptiX SDK.  This can be accomplished using `cmake-gui` or by entering the following command in a terminal window.  (Note that `..` specifies the path to the source code in the parent directory.)
```
cmake -DOptiX_INSTALL_DIR=/path/to/optix ..
```
Under Windows, it might be necessary to specify a generator and a toolset.  
```
cmake -G "Visual Studio 15 2017 Win64" -T host=x64 -DOptiX_INSTALL_DIR=/path/to/optix ..
```
- If the configuration is successful, build the OTK libraries.  Under Windows, simply load the Visual Studio solution file from the `build` directory.  Under Linux:
```
cd ..
make -j
```

If you encounter problems or if you have any questions, we encourage you to post on the [OptiX developer forum](https://forums.developer.nvidia.com/c/gaming-and-visualization-technologies/visualization/optix/167).

## Troubleshooting

Problem: CMake configuration error: "could not find git for clone of glad-populate" <br>
Solution: [git is required](https://git-scm.com/download) in order to download third party libraries (e.g. glad)

Problem: Runtime error: OPTIX_ERROR_UNSUPPORTED_ABI_VERSION: Optix call 'optixInit()' failed <br>
Solution: [Download newer driver](https://www.nvidia.com/download)

Problem: Windows Debug link error: cannot open file '..\zlib-install\lib\zlibstatic.lib' <br>
Solution: Build Release configuration first, then build Debug configuration.

Problem: CMake configuration error: "Failed to detect a default cuda architecture" <br>
Solution: Set configuration variable `CMAKE_CUDA_COMPILER` to the full path of the NVCC compiler.

## Attributions

This project contains build logic from the
[OptiX Wrapper Library (OWL)](https://github.com/owl-project/owl),
which is redistributed under the terms of the
[Apache License Version 2.0](https://github.com/owl-project/owl/blob/master/LICENSE).
