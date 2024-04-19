
# OptiX Toolkit Examples

Example programs demonstrating features in the OptiX Toolkit.

See the [CHANGELOG](CHANGELOG.md) for recent changes.

## Simple examples

- OtkHello, a simple "hello, world" example for the OptiX Toolkit.

## Demand Loading examples

- DemandLoadSimple shows basic demand loading of resources by page id.
- DemandLoadTexture demonstrates demand loaded textures on multiple GPUs.
- DemandGeometryViewer is a simple demand loaded geometry example.
- DemandPbrtScene uses demand geometry, demand materials and demand textures to render a PBRT scene.
- DemandTextureViewer visualizes texture tiles loaded from a mipmapped texture.
- TextureVariantViewer shows two textures with the same backing storage.
  One texture uses linear interpolation, and the other uses point interpolation.
- RayCones demonstrates using ray cones to drive texture filtering.
- udimTextureViewer demonstrates demand loaded udim textures.


## OmmBaking examples

- ommBakingSimple is a simple example of opacity micromap baking and creating an OptiX triangle
primitive acceleration structure using the micromap.  No rendering is performed.
- ommBakingViewer is an interactive viewer for a scene using opacity micromaps.

## Libraries

- pbrtParser wraps a pure virtual interface around the PBRT scene file parser from [pbrt v3](https://github.com/mmp/pbrt-v3).
- PbrtSceneLoader parses a PBRT scene file into a simple scene description.
