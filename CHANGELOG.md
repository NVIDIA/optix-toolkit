# OptiX Toolkit CMake changes

## Version 0.9.3

* embed_cuda now generates OptiX IR by default.  Be careful to use the PTX option when compiling pure CUDA kernels.

## Version 0.9.2

* Fixed intermittent PTX compilation issues caused by lack of trailing zero bytes in embedded PTX.

## Version 0.9

* `PrintTargetProperties` module added to aid in debugging CMake targets.
* vcpkg dependency support added for `FetchGtest` and `FetchOpenEXR` modules; when the dependency
  is found through vcpkg, not attempt is made to obtain these dependencies through FetchContent.
* OptiX find module updated to respect the requested version when multiple SDKs are installed.
* OptiX find module updated with reasonable default search locations for linux.
* `embed_ptx` updated to produce OptiX IR output and generated symbols are neutral in their
   identifiers and not tied to PTX or OptiX IR output options.  Error handling when running `bin2c`
   was improved.

## Version 0.8

* `embed_ptx` updated with additional keyword arguments.  See [embed_ptx.cmake](embed_ptx.cmake) for details.
