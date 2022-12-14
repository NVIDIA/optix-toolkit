# OptiX Toolkit ImageSource Library

The ImageSource library encapsulates the [OpenEXR](https://www.openexr.com/) image file reader.  It is used by the [DemandLoading](../DemandLoading/README.md)
library to read tiles from mipmapped images into sparse CUDA textures.  The library could easily be
adapted to support other tiled, mipmapped image formats such as [TIFF](http://www.libtiff.org/).
