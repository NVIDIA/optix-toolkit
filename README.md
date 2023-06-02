# OptiX Toolkit ShaderUtil Library

The OptiX Toolkit ShaderUtil is a library of functions commonly used in OptiX
and other raytracing programs.  It is currently header-only.

Among other things, the library provides utility headers for 
* [vector type math operations](include/OptiXToolkit/ShaderUtil/vec_math.h)
* [color representation and conversion operations](include/OptiXToolkit/ShaderUtil/color.h)
* [self-intersection avoidance for secondary rays](include/OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance.h)

## Ray cones for texture filtering

Helper functions in [ray_cone.h](include/OptiXToolkit/ShaderUtil/ray_cone.h) facilitate using ray
cones to drive texture filtering.  See the [Ray Cones whitepaper](docs/RayCones.pdf) for more information.
An [example application](https://github.com/NVIDIA/otk-examples/DemandLoading/RayCones) illustrates
how to use ray cones with the [Demand Loading library](https://github.com/NVIDIA/otk-demand-loading).

## Self-intersection avoidance

Ray and path tracing algorithms construct light paths by starting at the camera
or the light sources and intersecting rays with the scene geometry. As objects are
hit, new secondary rays are generated on these surfaces to continue the paths. In
theory, these secondary rays will not yield an intersection with the same surface
again, as intersections at a distance of zero are excluded by the intersection
algorithm. In practice, however, the finite floating-point precision used in the
actual implementation often leads to false positive results, known as
self-intersections, creating artifacts such as shadow acne, where the surface
sometimes improperly shadows itself.

OptiX Toolkit ShaderUtil provides a header-only library to compute spawn points for secondary
rays, safe from self-intersections. The spawn points are computed by offsetting along the surface
normal by a small but conservative epsilon. The library provides functionality to compute an object-space 
offset for Optix builtin triangles and to convert an object-space offset into world-space.
For custom primitives, developers should plug in a conservative object-space offset corresponding to their custom intersector.

The library provides both an OptiX API 
(See [OptixSelfIntersectionAvoidance.h](include/OptiXToolkit/ShaderUtil/OptixSelfIntersectionAvoidance.h)) 
and a CUDA API 
(See [CudaSelfIntersectionAvoidance.h](include/OptiXToolkit/ShaderUtil/CudaSelfIntersectionAvoidance.h))
for cuda/optix interoperability.

### Quick start

Here we show how to use the library to generate safe spawn points in an OptiX Closest-hit program.
The first step is to compute a surface point, normal and conservative offset in object-space.
The offset only needs to account for the object-space point construction and intersection.

```
float3 objPos, objNorm;
float objOffset;

if( optixIsTriangleHit() )
{
    SelfIntersectionAvoidance::getSafeTriangleSpawnOffset( objPos, objNorm, objOffset );
}
else
{
    // user implementation for custom primitives
    ...
}
```

The next step is to transform the object-space position, normal and offset into world-space.
The output world-space offset includes the input object-space offset and accounts for the transformation.

```
float3 wldPos, wldNorm;
float wldOffset;

SelfIntersectionAvoidance::transformSafeSpawnOffset( wldPos, wldNorm, wldOffset, objPos, objNorm, objOffset );
```

Finally, the offset is used to compute safe spawn points on the front and back of the surface.

```
float3 front, back;
SelfIntersectionAvoidance::offsetSpawnPoint( front, back, wldPos, wldNorm, wldOffset );
```

Secondary rays along the surface normal should use the generated front point as origin,
while rays pointing away from the normal should use the back point as origin.

```
float3 scatterPos = ( dot( scatterDir, wldNorm ) > 0.f ) ? front : back;
```

The above used functions apply to the hitpoint associated with the current intersection in Closest-hit and Any-hit programs.
The library provides similar functions for use in programs without an associated intersection (Such as Ray generation and callable programs).
