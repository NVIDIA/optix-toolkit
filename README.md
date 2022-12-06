
# OptiX Toolkit

A set of utilities commonly used in applications utilizing the [OptiX ray tracing API](https://developer.nvidia.com/rtx/ray-tracing/optix).

## Current Utilities
- **[PyOptiX](PyOptiX/README.md)** - Complete Python bindings for the OptiX host API.
- **[CuOmmBaking](CuOmmBaking/README.md)** - a C++/CUDA library for baking Opacity Micromap Arrays for textured geometry.
- **[DemandLoading](DemandLoading/README.md)** -  a C++/CUDA library for loading CUDA sparse textures on demand in OptiX renderers.
- **[ImageSource](ImageSource/README.md)** - wrapper for OpenEXR image library (adaptable to other image formats).
- **[Gui](Gui/README.md)** - convenience code for incorporating OpenGL into OptiX applications.
- **[ShaderOps](ShaderOps/README.md)** - vector math and other CUDA helper functions for OptiX kernels.
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

### Building statically linked libraries

OptiX Toolkit components are compiled into dynamic libraries (DSOs/DLLs) to simplify linking client
applications.  This eliminates the need for client applications to link with third-party libraries
like OpenEXR and GLFW.

Some clients of the OptiX Toolkit might prefer to use statically linked libraries.  This can be accomplished
by setting the CMake configuration variable `BUILD_SHARED_LIBS=OFF`.

Important: when building statically linked libraries, the CMake configuration variable
`OTK_FETCH_CONTENT` should be set to `OFF`, and various third party libraries must be installed as
described below.

### Installing third-party libraries.

Normally the OptiX Toolkit CMake files use `FetchContent` to download and build various third-party
libraries:
- Imath 3.1.5 or later
- OpenEXR 3.1.5 or later
- GLFW 3.3 or later
- glad (any recent version)

This behavior can be disabled by setting `OTK_FETCH_CONTENT=OFF` during CMake configuration,
which is necessary when building statically linked libraries, as described above.

When `FetchContent` is disabled, the following CMake configuration variables should be used to
specify the locations of the third-party libraries: `Imath_DIR`, `OpenEXR_DIR`, `glfw3_DIR`, and
`glad_DIR`.  The directory specified for each of these variables should be the location of the
project's CMake configuration file.  For example:
```
cd build
cmake \
-DBUILD_SHARED_LIBS=OFF \
-DOTK_FETCH_CONTENT=OFF \
-DImath_DIR=/usr/local/Imath/lib/cmake/Imath \
-DOpenEXR_DIR=/usr/local/OpenEXR/lib/cmake/OpenEXR \
-Dglfw3_DIR=/usr/local/glfw3/lib/cmake/glfw3 \
-Dglad_DIR=/usr/local/glad/lib/cmake/glad \
-DOptiX_ROOT_DIR=/usr/local/OptiX-SDK-7.5 \
..
```

As of this writing, OpenEXR 3.1 binaries are not yet available via most package managers, so we
recommend building these third party libraries from source code downloaded from the following
locations:

- Imath 3.1.5: https://github.com/AcademySoftwareFoundation/Imath.git
- OpenEXR 3.1.5: https://github.com/AcademySoftwareFoundation/openexr.git
- GLFW 3.3: https://github.com/glfw/glfw.git
- glad: https://github.com/Dav1dde/glad

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
