
# OptiX Toolkit (OTK) Examples

Example programs demonstrating features in the OptiX Toolkit.

See the [CHANGELOG](CHANGELOG.md) for recent changes.

## Current Examples

- OtkHello, a simple "hello, world" example for the OptiX Toolkit.
- Demand Loading:
  - demandLoadSimple shows basic demand loading of resources by page id.
  - demandLoadTexture demonstrates of demand loaded textures on multiple GPUs.
  - DemandGeometryViewer is a simple demand loaded geometry example.
  - demandTextureViewer visualizes texture tiles loaded from a mipmapped texture.
  - textureVariantViewer shows two textures with the same backing storage.
  One texture uses linear interpolation, and the other uses point interpolation.
  - udimTextureViewer demonstrates demand loaded udim textures.
- CmOmmBaking:
  - ommBakingSimple is a simple example of opacity micromap baking and creating an OptiX triangle
primitive acceleration structure using the micromap.  No rendering is performed.
  - ommBakingViewer is an interactive viewer for a scene using opacity micromaps.
  