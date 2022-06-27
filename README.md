
# OptiX Utility Toolkit

The OptiX Utility Toolkit (OTK) is a collection of libraries and utilities that simplify building
applications with the [OptiX SDK](https://developer.nvidia.com/designworks/optix/download).  It includes
the following components:

- [PyOptiX](PyOptiX/README.md): Python bindings for OptiX 7 API calls.
- [DemandLoading](DemandLoading/README.md): a C++/CUDA library for loading CUDA sparse textures on demand in OptiX renderers.
- OtkGui: convenience code for incorporating OpenGL into OptiX applications.
- OtkCuda: vector math and other CUDA helper functions for OptiX kernels.
- OtkUtil: file handling and other utility functions.

# Requirements

- OptiX 7.4 or later.
- CUDA 11.1 or later.
- C++ compiler (e.g. gcc under Linux, Visual Studio under Windows)
- CMake 3.20 or later.  Using the latest CMake is highly recommended, to ensure up-to-date CUDA
language support.

# Building OTK

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

If you encounter problems or if you have any questions, we encourage you to post on the OptiX forums:
https://devtalk.nvidia.com/default/board/90/
