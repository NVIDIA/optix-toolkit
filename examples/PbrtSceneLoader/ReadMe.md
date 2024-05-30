# PbrtSceneLoader Library

This library implements the `otk::pbrt::Api` interface provided by the `pbrtParser` library
and provides the semantics of the keywords in pbrt scene file.  This library was created to
support the `DemandPbrtScene` example and makes some assumptions that may not be applicable
to other examples.  In particular, a limited number of shapes are implemented and simplistic
lighting and material models are used.

The `SceneLoader` interface can load from a file or from a string.  The string loading is used
in unit tests to validate the semantics of the scene loader.  The file loading is used in the
`DemandPbrtScene` example.

The `SceneLoader` implementation depends on an instance of a `otk::pbrt::Logger` and an
instance of a `otk::pbrt::MeshReader`.  Instances of these interfaces can be created with
the classes declared in `<OptiXToolkit/PbrtSceneLoader/GoogleLogger.h>` and
`<OptiXToolkit/PbrtSceneLoader/PlyReader.h>`, respectively.

The scene description, declared in `<OptiXToolkit/PbrtSceneLoader/SceneDescription.h>`, is
implemented in terms of pbrt's data types, e.g. `pbrt::Point3f`.  Inline triangle meshes are
read directly into memory during parse, whereas PLY meshes are scanned to compute a bounds and
associated with an `otk::pbrt::MeshLoader` interface to support delay loading of PLY meshes
into host memory.
