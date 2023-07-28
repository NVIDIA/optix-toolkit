
# OptiX Toolkit (OTK)

A set of utilities commonly used in applications utilizing the [OptiX ray tracing API](https://developer.nvidia.com/rtx/ray-tracing/optix).

See the [CHANGELOG](CHANGELOG.md) for recent changes.

## Current Utilities
- **[DemandLoading](https://github.com/NVIDIA/otk-demand-loading)** -  a C++/CUDA library for loading CUDA sparse textures on demand in OptiX renderers.
- **[Memory](https://github.com/NVIDIA/otk-memory)** - Memory allocators (used by DemandLoading library).
- **[OmmBaking](https://github.com/NVIDIA/otk-omm-baking)** - a C++/CUDA library for baking Opacity Micromap Arrays for textured geometry.
- **[PyOptiX](https://github.com/NVIDIA/otk-pyoptix)** - Complete Python bindings for the OptiX host API.
- **[ShaderUtil](https://github.com/NVIDIA/otk-shader-util)** - Header-only libraries for OptiX kernels (e.g. vector math, Self Intersection Avoidance).

Each of these components is stored in a separate git repository, which is referenced as a git submodule.
After checking out the OptiX Toolkit repository, be sure to initialize the submodules, e.g.
```
git submodule update --init --recursive
```
Alternatively, a subset of the submodules can be specified.  For example, the DemandLoading library
can be built in isolation by initializing its submodule, along with the submodules upon which it depends:
```
git submodule update --init --recursive DemandLoading Memory ShaderUtil CMake
```

## Requirements

- OptiX 7.3 or later.
- CUDA 11.1 or later.
- C++ compiler (gcc, Clang, or Visual Studio).
- CMake 3.24 or later.  Using the latest CMake is recommended to ensure up-to-date CUDA language support.
- git (any modern version with LFS).

On some Linux systems it may be necessary to install some commonly used developer packages with the following commands:
```
sudo apt-get install curl git-lfs pkg-config
git lfs install
```
Building the OptiX Toolkit examples requires an OpenGL development environment.  On most Linux systems the necessary packages can be installed using the following command:
```
sudo apt-get install libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev mesa-common-dev 
```

### Third-party libraries

The OptiX Toolkit build system employs a "download if missing" workflow.  The following third-party libraries
are downloaded and built if necessary:
- Imath 3.1.5 or later
- OpenEXR 3.1.5 or later
- GLFW 3.3 or later
- glad (any recent version)

The DemandLoading library can optionally employ OpenImageIO (if installed) to read image files.
However, OpenImageIO is not downloaded if missing.

### Using vcpkg

Alternatively, [vcpkg](https://github.com/microsoft/vcpkg) is a convenient way to provide the third-party libraries required by OTK.
First, create a local `vcpkg` repository as follows:
```
git clone https://github.com/microsoft/vcpkg
./vcpkg/bootstrap-vcpkg
./vcpkg/vcpkg integrate install
```
The final step will print the path to a "toolchain file" that should be provided during CMake configuration, e.g.
```
cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg.cmake [...]
```
OTK provides a `vcpkg` manifest (called [vcpkg.json](https://github.com/NVIDIA/optix-toolkit/blob/master/vcpkg.json))
that specifies which third-party libraries are required.  CMake invokes `vcpkg` to download and
build these packages during configuration.

Tip: if you use an existing `vcpkg` repository, be sure to update it (using `git pull`) before configuring OTK to ensure 
that up-to-date versions of third-party libraries are used (particularly OpenEXR).

## Building the OptiX Toolkit

- In the directory containing the OTK source code, create a subdirectory called `build` and `cd` to that directory.
```
mkdir build
cd build
```
- Configure CMake, specifying the location of the OptiX SDK and (optionally) the `vcpkg` toolchain file.  This can be accomplished using `cmake-gui` or by entering the following command in a terminal window.  (Note that `..` specifies the path to the source code in the parent directory.)
```
cmake -DOptiX_INSTALL_DIR=/path/to/optix -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg.cmake ..
```
Under Windows, be sure to use a 
[Visual Studio developer console](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell)
or use the `vcvarsall.bat` batch file to set up necessary environment variables.  It might also be
necessary to specify a generator and a toolset under Windows:
```
cmake -G "Visual Studio 15 2017 Win64" -T host=x64 -DOptiX_INSTALL_DIR=/path/to/optix -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg.cmake ..
```
- If the configuration is successful, build the OTK libraries.
  * Under Windows, simply load the Visual Studio solution file from the `build` directory.  
  * Under Linux, run `make -j` in the `build` directory.
- Alternatively, building with [Ninja](https://ninja-build.org/) works well under both Linux and Windows.
```
cmake -G Ninja [...]
ninja
```

If you encounter problems or if you have any questions, we encourage you to post on the [OptiX developer forum](https://forums.developer.nvidia.com/c/gaming-and-visualization-technologies/visualization/optix/167).

### Disabling FetchContent

The OptiX Toolkit uses CMake's `FetchContent` feature to download and build any missing third-party libraries.
The use of `FetchContent` can be disabled by setting `OTK_FETCH_CONTENT=OFF` during CMake configuration,
which is necessary when building statically linked libraries, as described below.

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
When `FetchContent` is disabled, using `vcpkg` as described above is recommended.  Alternatively,
the necessary third-party libraries from source code downloaded from the following locations:

- Imath 3.1.5: https://github.com/AcademySoftwareFoundation/Imath.git
- OpenEXR 3.1.5: https://github.com/AcademySoftwareFoundation/openexr.git
- GLFW 3.3: https://github.com/glfw/glfw.git
- glad: https://github.com/Dav1dde/glad

### Building statically linked libraries

OptiX Toolkit components are compiled into dynamic libraries (DSOs/DLLs) to simplify linking client
applications.  This eliminates the need for client applications to link with third-party libraries
like OpenEXR and GLFW.

Some clients of the OptiX Toolkit might prefer to use statically linked libraries.  This can be accomplished
by setting the CMake configuration variable `BUILD_SHARED_LIBS=OFF`.

Important: when building statically linked libraries, the CMake configuration variable
`OTK_FETCH_CONTENT` should be set to `OFF`, and various third-party libraries must be installed as
described above.

## Troubleshooting

Problem: CMake Error: include could not find requested file: Policies
Solution: Git submodules must be initialized, e.g. `git submodule update --init --recursive`

Problem: add_library cannot create ALIAS target "OpenEXR::Config" because another target with the same name already exists.
Solution: Install OpenEXR 3.1 or later or set `OpenEXR_DIR` to such an installation.

Problem: CMake configuration error: "could not find git for clone of glad-populate" <br>
Solution: [git is required](https://git-scm.com/download) in order to download third-party libraries (e.g. glad)

Problem: Runtime error: OPTIX_ERROR_UNSUPPORTED_ABI_VERSION: Optix call 'optixInit()' failed <br>
Solution: [Download newer driver](https://www.nvidia.com/download)

Problem: CMake configuration error: "Failed to detect a default cuda architecture" <br>
Solution: Set configuration variable `CMAKE_CUDA_COMPILER` to the full path of the NVCC compiler.

## Attributions

This project contains build logic from the
[OptiX Wrapper Library (OWL)](https://github.com/owl-project/owl),
which is redistributed under the terms of the
[Apache License Version 2.0](https://github.com/owl-project/owl/blob/master/LICENSE).
