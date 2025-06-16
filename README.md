# OptiX Toolkit (OTK)

A set of utilities commonly used in applications utilizing the [OptiX ray tracing API](https://developer.nvidia.com/rtx/ray-tracing/optix).

See the [CHANGELOG](CHANGELOG.md) for recent changes.

After checking out the toolkit, be sure to initialize the submodules before building:
```
git submodule update --init --recursive
```

## Submodules merged in v1.0.0

If you cloned the OptiX Tookit repository prior to v1.0.0, we recommend cloning a fresh repository.
We restructured the repository, merging multiple submodules into the main repository, which
complicates performing a pull operation.  (If you are intent on pulling, use `git submodule deinit`
on all of the submodules except `vcpkg` before pulling.)


## Current Utilities
- **[DemandLoading](DemandLoading)** -  a C++/CUDA library for loading CUDA sparse textures on demand in OptiX renderers.
- **[Memory](Memory)** - Memory allocators (used by DemandLoading library).
- **[OmmBaking](OmmBaking)** - a C++/CUDA library for baking Opacity Micromap Arrays for textured geometry.
- **[PyOptiX](PyOptiX)** - Complete Python bindings for the OptiX host API.
- **[ShaderUtil](ShaderUtil)** - Header-only libraries for OptiX kernels (e.g. vector math, Self Intersection Avoidance).

## Requirements

- C++ compiler (gcc, Clang, or Visual Studio)
- git (any modern version with LFS) ([download](https://git-scm.com/downloads))
- CMake 3.27 or later ([download](https://cmake.org/download/)).  
  - Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
- CUDA 11.1 or later ([download](https://developer.nvidia.com/cuda-downloads))
- OptiX 7.3 or later ([download](https://developer.nvidia.com/designworks/optix/download))
- Other third-party libraries are downloaded and built on demand (see below).

On some Linux systems it may be necessary to install some commonly used developer packages with the following commands:
```
sudo apt-get install curl git-lfs pkg-config
git lfs install
```
Building the examples included with the OptiX Toolkit requires an OpenGL development environment.
On most Linux systems the necessary packages can be installed using the following command:
```
sudo apt-get install libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev mesa-common-dev
```
Under Rocky linux, the following packages must be installed to build the GL-based OTK examples:
```
sudo dnf install libXcursor-devel libXi-devel libXinerama-devel libXrandr-devel mesa-libGLU-devel pkg-config
```

## Building the OptiX Toolkit with the Supplied CMake Presets

[CMake presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) provide a convenient
means of specifying CMake configure settings, build settings and test settings as a named collection.
A `CMakePresets.json` file is provided with the toolkit to cover some basic use cases and
for use as building blocks for creating your own preset that covers your specific use case.

To build the toolkit with default settings, issue the following command from the source code
directory.

```
cmake --preset otk-default
cmake --build --preset otk-default -j
```

The following command runs the tests using the default preset.  (Note that `ctest` requires a value for the `-j` option.)
```
ctest --preset otk-default -j 16
```

All supplied presets begin with the prefix `otk-` so that they won't conflict with your personal
presets.  It is recommended that you store your personal presets in the file `CMakeUserPresets.json`
in the source directory so that future updates to the toolkit won't conflict with your
personal presets.

The supplied presets create build directories as children of the source directory with the name
of the preset in the build directory name, e.g. `build-otk-default`.  This gives each preset a
distinct build directory allowing you to experiment with different presets without them interfering
with one another.

Consult the `CMakePresets.json` file for the available building blocks for use in creating your
own customized preset that suits your individual needs.

## Building the OptiX Toolkit Manually

Building the toolkit follows the standard CMake worfklow: configure, build and test.

- Create a directory called `build` and `cd` to that directory.
  ```
  mkdir build
  cd build
  ```
- Configure the toolkit using CMake.
  This can be accomplished using the [CMake GUI tool](https://cmake.org/cmake/help/latest/manual/cmake-gui.1.html),
  the [CMake console tool](https://cmake.org/cmake/help/latest/manual/ccmake.1.html) (not available on Windows as of this writing),
  or from the command-line directly by entering the following command:
  (Note that `..` specifies the path to the source code from the build directory.)
  ```
  cmake ..
  ```
  This will configure the toolkit with the default options create a build project using
  the default CMake generator for your platform.
- If the configuration is successful, build the OTK libraries with the following command:
  `cmake --build . --config Release`
- After building you can execute the tests with the following command:
  `ctest -C Release`

If you wish to customize the build of the toolkit, see the section on [options](README.md#optix-toolkit-options).

If you encounter problems or if you have any questions, we encourage you to post on the
[OptiX developer forum](https://forums.developer.nvidia.com/c/gaming-and-visualization-technologies/visualization/optix/167).

## OptiX Toolkit Options

The following options may be supplied to CMake at configure time to customize the toolkit:

Variable | Type | Default | Description
-------- | ---- | ------- | -----------
`OTK_USE_VCPKG` | `BOOL` | `ON` | Use [vcpkg](https://vcpkg.io/) for [dependencies](README.md#third-party-libraries).
`OTK_USE_VCPKG_OPENEXR` | `BOOL` | `${OTK_USE_VCPKG}` | Obtain OpenEXR via vcpkg.
`OTK_USE_OIIO` | `BOOL` | `OFF` | Use [OpenImageIO](https://openimageio.readthedocs.io/) to read PNG and JPEG files as image sources.
`OTK_FETCH_CONTENT` | `BOOL` | `ON` | Use [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html) for [dependencies](README.md#third-party-libraries) if `OTK_USE_VCPKG` is `OFF`.
`OTK_BUILD_EXAMPLES` | `BOOL` | `ON` | Build the examples.
`OTK_BUILD_TESTS` | `BOOL` | `ON` | Build the tests.
`OTK_BUILD_DOCS` | `BOOL` | `ON` | Build the doxygen documentation.
`OTK_BUILD_PYOPTIX` | `BOOL` | `OFF` | Build the PyOptiX python module.
`OTK_PROJECT_NAME` | `STRING` | `OptiXToolkit` | Project name for the generated build scripts.
`OTK_LIBRARIES` | `STRING` | `ALL` | List of libraries to build.

If both `OTK_USE_VCPKG` and `OTK_FETCH_CONTENT` are `ON`, vcpkg will be used for dependencies.

If the option `OTK_LIBRARIES` is used to configure the libraries to build, the value should be a semi-colon
separated list of one or more of the names `DemandLoading`, `Memory`, `OmmBaking` or `ShaderUtil`.
The default value `ALL` is the same as specifying `DemandLoading;Memory;OmmBaking;ShaderUtil`.
Some libraries depend on other libraries.  The CMake build script includes dependent libraries as needed.

## Third-party Libraries

The toolkit depends on third party libraries for texture file format parsing, a GUI framework for
the examples and a unit test framework for the tests.  

| Component         | Dependency                |
| ----------------- | ------------------------- |
| **DemandLoading** | Imath 3.1.5 or later      |
|                   | OpenEXR 3.1.5 or later    |
|                   | (optional) OpenImageIO    |
| **Examples**      | imgui                     |
|                   | GLFW 3.3 or later         |
|                   | glad (any recent version) |
|                   | stb                       |
| **Tests**         | gtest                     |

The toolkit can automatically obtain these third party libraries in one of two ways: via a [vcpkg](README.md#vcpkg)
manifest or via [FetchContent](README.md#fetchcontent) as described below.

Using vcpkg is the recommended method of obtaining third-party libraries.

### VcPkg

The toolkit repository contains a [vcpkg](https://vcpkg.io) manifest to download and build
third party libraries.  The repository includes the `vcpkg` repository as a submodule to select
the specific versions of dependencies used.  A vcpkg manifest, `vcpkg.json`, specifies
the dependencies to be used.  The vcpkg standard CMake integration via a toolchain file
is used to bootstrap vcpkg and obtain the dependencies.

The use of `vcpkg` can be disabled by configuring with `OTK_USE_VCPKG=OFF`, which will
cause the toolkit to use [FetchContent](README.md#fetchcontent) for third party libraries.

When the toolkit is used as a subdirectory, e.g. a git submodule, of another project,
the vcpkg manifest for toolkit must be incorporated into the parent project's manifest
for vcpkg to correctly fetch the third party libraries used by the toolkit.

The CMake module `ProjectOptions` (from the toolkit's `CMake` directory) should be included by
the parent project's `CMakeLists.txt` before the first call to [`project`](https://cmake.org/cmake/help/latest/command/project.html).
This gives the toolkit the chance to configure optional features from the manifest
and configure options controlling how the toolkit is built.

If the variable `CMAKE_TOOLCHAIN_FILE` is not set when `ProjectOptions` is included, it will
be set to point to the CMake integration in the toolkit's `vcpkg` submodule.  A parent
project using vcpkg for dependency management may be using its own submodule of vcpkg for
toolchain integration and should set `CMAKE_TOOLCHAIN_FILE` as appropriate before including
the toolkit's `ProjectOptions` module.

### FetchContent

If `vcpkg` is disabled, the toolkit will use CMake's `FetchContent` feature to download and build
any missing third-party libraries.  The use of `FetchContent` can be disabled by setting
`OTK_FETCH_CONTENT=OFF` during CMake configuration, which is necessary when building statically
linked libraries, as described below.

When `FetchContent` is disabled, the following CMake configuration variables should be used to
specify the locations of the third-party libraries: `Imath_DIR`, `OpenEXR_DIR`, `glfw3_DIR`, and
`glad_DIR`.  The directory specified for each of these variables should be the location of the
project's CMake configuration file.  For example:
```
cd build
cmake \
-DBUILD_SHARED_LIBS:BOOL=OFF \
-DOTK_FETCH_CONTENT:BOOL=OFF \
-DOTK_BUILD_EXAMPLES:BOOL=OFF \
-DOTK_BUILD_TESTS:BOOL=OFF \
-DImath_DIR:PATH=/usr/local/Imath/lib/cmake/Imath \
-DOpenEXR_DIR:PATH=/usr/local/OpenEXR/lib/cmake/OpenEXR \
-Dglfw3_DIR:PATH=/usr/local/glfw3/lib/cmake/glfw3 \
-Dglad_DIR:PATH=/usr/local/glad/lib/cmake/glad \
../optix-toolkit
```
When `FetchContent` is disabled, using `vcpkg` as described above is recommended.  Alternatively,
the necessary third-party libraries from source code downloaded from the following locations:

- Imath 3.1.5: https://github.com/AcademySoftwareFoundation/Imath.git
- OpenEXR 3.1.5: https://github.com/AcademySoftwareFoundation/openexr.git
- GLFW 3.3: https://github.com/glfw/glfw.git
- glad: https://github.com/Dav1dde/glad

### Building Statically Linked Libraries

OptiX Toolkit components are compiled into dynamic libraries (DSOs/DLLs) to simplify linking client
applications.  This eliminates the need for client applications to link with third-party libraries
like OpenEXR and GLFW.

Some clients of the toolkit might prefer to use statically linked libraries.  This can be accomplished
by setting the CMake configuration variable `BUILD_SHARED_LIBS=OFF`.

Important: when building statically linked libraries, the CMake configuration variable
`OTK_FETCH_CONTENT` should be set to `OFF`, and various third-party libraries must be installed as
described above.

## Troubleshooting

**Problem:** CMake configuration error: "`OTK_USE_VCPKG` is ON, but could not locate vcpkg toolchain file"<br>
**Solution:** vcpkg submodule must be initialized, e.g. `git submodule update --init --recursive`

**Problem:** add_library cannot create ALIAS target "OpenEXR::Config" because another target with the same name already exists.<br>
**Solution:** Install OpenEXR 3.1 or later or set `OpenEXR_DIR` to such an installation.

**Problem:** CMake configuration error: "could not find git for clone of glad-populate" <br>
**Solution:** [git is required](https://git-scm.com/download) in order to download third party libraries (e.g. glad)

**Problem:** Runtime error: `OPTIX_ERROR_UNSUPPORTED_ABI_VERSION`: Optix call 'optixInit()' failed <br>
**Solution:** [Download newer driver](https://www.nvidia.com/download)

**Problem:** CMake configuration error: "Failed to detect a default cuda architecture" <br>
**Solution:** Set configuration variable `CMAKE_CUDA_COMPILER` to the full path of the NVCC compiler.

**Problem:** Tests fail with "Cannot read image file filename.exr. File is not an image file."
**Solution:** Install git lfs (`git lfs install`) and re-clone repository.

**Problem:** Compiling the CUDA compiler identification source file failed.  gcc versions later than 13 are not supported.
**Solution:** Install gcc-13 (`sudo apt-get install gcc-13`) and use `update-alternatives` to make gcc-13 the default.

**Problem:** Running vcpkg install - failed
**Solution:** `git submodule deinit vcpkg; git submodule update --init vcpkg`

If you encounter a problem, we encourage you to post on the [OptiX forums](https://devtalk.nvidia.com/default/board/90/) or open a ticket on the [OptiX Toolkit issues](https://github.com/NVIDIA/optix-toolkit/issues) page on GitHub.

## Attributions

This project contains build logic from the
[OptiX Wrapper Library (OWL)](https://github.com/owl-project/owl),
which is redistributed under the terms of the
[Apache License Version 2.0](https://github.com/owl-project/owl/blob/master/LICENSE).



