# DemandPbrtScene Example

This example parses a scene file created for Matt Pharr's Physically Based Ray Tracer,
[pbrt v3](https://github.com/mmp/pbrt-v3), and uses demand geometry, demand materials
and demand textures to render the scene.  It demonstrates demand loading of assets in
a real-world use case.  The example does not attempt to implement the full semantics
of the sampling, lighting and material models in pbrt, so the rendering should only
be considered approximate with respect to the output of the pbrt renderer.

Where practical, all of the code was written test-first using the Google Test and
Google Mock for unit tests and rendering to a file with an image comparison for integration
tests.  To support testing, all of the implementation code is in the library
`DemandPbrtSceneImpl` and the executable target simply consists of a delegating
implementation of `main`.  The tests are located in the `tests` subdirectory and
the `TestDemandPbrtSceneImpl` target.

The application is divided primarily into four classes:

- `Application` acts as a container for all the dependencies and implements the main
  rendering loop.
- `UserInterface` is responsible for all interactive manipulation of the rendering with
  the mouse and keyboard and displays the interactive result.  The user interface is not
  instantiated when rendering to a file.
- `Scene` is responsible for managing all of the resources associated with the scene and
  performs all demand loading of assets.  The scene creates all OptiX resources that are
  associated with geometry and programs.
- `Renderer` manages the OptiX launch and associated resources such as the OptiX pipeline
  and shader binding table.

`UserInterface`, `Scene` and `Renderer` are abstract interfaces from the point of view of
the `Application` and are implemented by the classes `ImGuiUserInterface`, `PbrtScene` and
`OptixRenderer`.

Command-line options, including a required pbrt scene filename, are parsed into an `Options`
structure owned by the `Application`.  (Run with `--help` to see a list of all the options.)
`OptixRenderer` and `PbrtScene` are observers of changes to the options and the
`ImGuiUserInterface` can modify the options in response to user input.

## PbrtScene Class

All of the demand loading mechanisms are in the `PbrtScene` class, whereas the other classes
contain code that is similar across many OptiX example programs.  When initialized, `PbrtScene`
will parse the scene file into a `SceneDescription` using the `PbrtSceneLoader` library.
A demand loaded geometry proxy is created via the `DemandGeometry` library to represent the
bounds of the entire scene.  The necessary OptiX resources, such as an instance acceleration
structure `optixTraversableHandle`, `optixModule`s and `optixProgramGroup`s to render proxy
geometries and proxy materials are created.

### Rendering to Resolve Proxies on Demand

During an `optixLaunch`, proxy geometries are intersected.  This results in a requested id
coming back to the application from the `DemandGeometry` library.  Proxy geometries that have
been previously loaded as realized geometry with a proxy material may have been intersected and
the requested proxy material ids come back to the application from the `DemandMaterial` library.
Realized geometry with realized materials may be associated with a diffuse texture; the
texture is demand loaded based on requested intersection points.  Loading of the requested
texture tiles is handled transparently by the `DemandLoading` library.

Pbrt has a notion of an "alpha cut out" texture that gives a boolean indicator of transparency
at a particular surface intersection location.  (Partial transparency in pbrt is handled
through the material mechanism and is not implemented by this sample.)  For materials with
an alpha cut out texture, a two-phase proxy material resolution mechanism is employed.  First,
the realized geometry -- with UV coordinates -- is associated with a proxy material and a
demand loaded alpha cut out texture.  The cut out texture is sampled in the Any Hit program
and if intersected, the associated demand material id will be reported back to the application
after the launch completes.  Such requested materials will then be completely resolved into
a fully realized material, usually with a demand loaded diffuse texture map.

The options `--oneshot-geometry` and `--oneshot-material` can be used to enable resolution
of proxy geometries and proxy materials one at a time via the `G` and `M` keys.  This allows
the user to visualize the resolution process interactively.

A client of the `DemandGeometry` and `DemandMaterial` libraries can choose how to visualize
intersected proxies in the rendered image.  This example picks intentionally bright, ugly
colors for intersected proxies so that they stand out in the rendered scene and aren't mistaken
for final rendered pixels.

## Caching and Sharing Data

The example attempts to avoid loading the same data into the GPU twice by using caches.
`ImageSource` instances created from files and adapted image sources for use as alpha cut out
textures, diffuse textures or sky box textures are all cached separately.  The sources
associated with files may be used by the scene description as both an alpha cut out texture
and a diffuse texture.  The same image source will be adapted for two different uses, with
the two adapted image sources sharing the same underlying image source.

Acceleration structures and associated per-vertex data for meshes loaded from PLY files are
cached based on the filename.  (Pbrt parsing always returns absolute paths for resolved
filenames.)  When multiple instances of an object in a pbrt scene occur and they reference
the same PLY file, only a single geometry acceleration structure is used for the underlying
mesh.  No attempt was made to identify duplicate inline triangle meshes.

## Scene Graph Organization

The sample supports two methods of scene organization into acceleration structures.  The first
creates acceleration structures that correspond directly to the `Shape` elements in the pbrt
scene description.  Each `Shape` corresponds one-to-one with a geometry acceleration structure
(GAS).  In this method, a pbrt `Object` is one or more GASes, one per `Shape` in the `Object`.
This is the default scene graph organization and can be explicitly selected with the command-line
argument `--proxy-granularity=fine`.

The second method groups as many `Shape` elements for an `Object` as possible into as few
GASes as necessary to represent the object.  There are some constraints that prevent combining
different shapes together into a single GAS.  For instance, a GAS must contain primitives all
of the same type, so a sphere shape cannot be combined with a triangle mesh shape.  All primitives
in a GAS must share the same set of OptiX programs, so if some triangles use an alpha cutout texture
map and some triangles do not, then they will belong to separate GASes.  This method of scene
organization can be selected with the `--proxy-granularity=coarse` command-line argument.

There is no support for switching the scene graph organization dynamically.

## Launch Parameter Scene Data

The OptiX programs used by the sample use the launch parameters structure, `Params`, to access
all necessary secondary data not stored in the GASes.  This includes per-vertex data such as
normals and texture coordinates as well as all material data.

All GASes are held inside an Instance Acceleration Strucure (IAS) and associated with an
instance id.  The instance id is used as the primary index into the following arrays:

- `instanceNormals`
- `instanceUVs`
- `partialUVs`
- `partialMaterials`
- `materialIndices`

The `Normals` and `UVs` arrays hold pointers to arrays of the additional per-vertex data
associated with the GAS.  The per-vertex data arrays are indexed by the primitive index within
the GAS to obtain the necessary data.

The `partialMaterials` array is used to obtain the alpha cutout map texture id for geometry
resolved to an alpha cutout map, but not yet fully intersected.  If the cutout is such that
rays never intersect the actual surface, then materials and textures for the surface are
never loaded.

The `materialIndices` array is used to find the number of material groups associated with
a GAS.  Each entry in the array gives the number of material groups in the GAS and the starting
index into the `primitiveMaterials` array.  The `primitiveMaterials` array contains one
`PrimitiveMaterialRange` entry for each material group in the GAS.  Each `PrimitiveMaterialRange`
entry gives a range of primitive indices in the GAS associated with a material index.

The `realizedMaterials` array, indexed by a material id, holds the material parameters
for the simple material model used in the sample.

![Params](Params.png)
