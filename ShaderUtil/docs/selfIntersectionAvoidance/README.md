# Self-intersection avoidance

Ray and path tracing algorithms construct light paths by starting at the camera or the light sources and intersecting rays with the scene geometry. As objects are hit, new secondary rays are generated on these surfaces to continue the paths. In theory, these secondary rays will not yield an intersection with the same surface again, as intersections at a distance of zero are excluded by the intersection algorithm. In practice, however, the finite floating-point precision used in the implementation often leads to false positive results, known as self-intersections, creating artifacts such as shadow acne, where the surface sometimes improperly shadows itself.

OptiX Toolkit ShaderUtil provides a header-only library to compute spawn points for secondary rays, safe from self-intersections. The spawn points are computed by offsetting along the surface normal by a small but conservative epsilon. The library provides functionality to compute an object-space offset for OptiX builtin triangles and to convert an object-space offset into world-space. For custom primitives, developers should plug in a conservative object-space offset corresponding to their custom intersector.

The library provides both OptiX and CUDA APIs:

* OptiX: [OptixSelfIntersectionAvoidance.h](include/OptiXToolkit/ShaderUtil/OptixSelfIntersectionAvoidance.h) 
* CUDA: [CudaSelfIntersectionAvoidance.h](include/OptiXToolkit/ShaderUtil/CudaSelfIntersectionAvoidance.h)


### Quick start

Here we show how to use the library to generate safe spawn points in an OptiX Closest-hit program. The first step is to compute a surface point, normal and conservative offset in object-space. The offset only needs to account for the object-space point construction and intersection.

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

The next step is to transform the object-space position, normal and offset into world-space. The output world-space offset includes the input object-space offset and accounts for the transformation.

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

The above used functions apply to the hitpoint associated with the current intersection in Closest-hit and Any-hit programs. The library provides similar functions for use in programs without an associated intersection (Such as Ray generation and callable programs).
