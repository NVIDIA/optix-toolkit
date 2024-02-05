# OptiX Toolkit Examples changes

## Version 0.9

* The Demand Geometry Viewer sample was updated with one-shot resolution of proxy geometries and
  materials.  The Demand Material library is now used by the sample.
* The `stb` library details were moved to the FetchStb CMake module and vcpkg support was added.
* Added vcpkg support for obtaining the `imgui`, `glfw` and `glad` dependencies.
* The Demand Texture Viewer sample updated to load any image source and not just EXR files.
  Command-line options added to apply mipmap and tile adapters to the loaded image source.
* The ImageSource CMake module was updated to create a 'resource' target associated with the
  target for which image tests are being run.  This keeps gold images and resource files in the binary
  directory up to date if they change in the source directory.  A FOLDER parameter was added to specify
  the FOLDER property for the resource target.  A RESOURCES parameter was added to specify additional
  resource files, besides the implied gold image for the test, to be copied to the binary directory.
  This is usually used to copy additional program inputs such as scene files.  Failure output was
  improved for readability.

## Version 0.8.1

* Added [ray cones](DemandLoading/RayCones) example.
* Added [texture painting](DemandLoading/TexturePainting) example.
  * Known issue: the texture painting example crashes when multiple GPUs are enabled.

## Version 0.8

* Added [DemandGeometryViewer](DemandLoading/DemandGeometryViewer) example.
