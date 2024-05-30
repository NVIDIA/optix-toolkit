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
