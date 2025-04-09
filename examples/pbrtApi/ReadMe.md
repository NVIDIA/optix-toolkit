# pbrtParser Library

This library uses the source code from Matt Pharr's Physically Based Ray Tracer,
[pbrt v3](https://github.com/mmp/pbrt-v3), to create a parser for pbrt scene files that
invokes methods on a pure virtual interface. No semantics are imposed, only file parsing
is performed.  Using the pbrt code directly ensures that the parse happens exactly as pbrt
would parse.

The pbrt parser assumes that there are functions in the `pbrt` namespace that it can call
for each parsed node, such as:

```
namespace pbrt {

// ...
void pbrtObjectBegin( const std::string& name );
// ...

} // namespace pbrt
```

This library implements those global functions by delegating to a corresponding method
on the `otk::pbrt::Api` interface:

```
namespace otk { namespace pbrt {

class Api
{
  public:
    // ...
    virtual void objectBegin( const std::string& name ) = 0;
    // ...
};

} // namespace pbrt
} // namespace otk
```

A client of this library would implement the pure virtual interface in
`<OptiXToolkit/PbrtApi/PbrtApi.h>` and call the function `otk::pbrt::setApi` to set the
single active instance of this interface.

It is the responsibility of the `otk::pbrt::Api` implementation to provide the appropriate
semantics for a pbrt scene.  For instance there is a current transformation matrix 
stack that is manipulated by the `AttributeBegin` and `AttributeEnd` keywords in the scene
file.  This library provides no implementation of any pbrt semantics, only delegation to
the abstract interface.

See the PbrtSceneLoader library for an example of an implementation of `otk::pbrt::Api`
that implements the semantics of the pbrt scene description.

The pbrt source code depends on a specific version of the google log library and any
implementation of the `Api` interface must properly initialize google log before parsing.
