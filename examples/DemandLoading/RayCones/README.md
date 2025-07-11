# RayCones example

[Ray cones](https://github.com/NVIDIA/optix-toolkit/tree/master/ShaderUtil/docs/rayCones) allow a path tracer to track the footprint of a ray sample as the ray propagates. The *rayCones* example shows how to drive the OTK demand texture system in a path tracer using ray cones.

## Quick Start

The OTK `RayCone` class can track a ray's footprint as it propagates in a path tracer. To track a ray's footprint, first construct a ray cone for the eye ray in the OptiX *raygen* program. Below is an example of how to do this for a thin lens camera:

```
RayCone rayCone = initRayConeThinLensCamera( lookAt-eye, lensWidth, rayDirection );
```

After a ray has been cast, propagate the ray cone to the intersection point based on the ray distance. 

```
rayCone = propagateRayCone( rayCone, distance );
```

When tracing secondary rays the ray cone must be transformed at the surface based whether the scattered ray is a reflection or transmission, the curvature of the surface, and the BSDF value at the intersection point. For example, reflection code might look like this:

```
rayCone = reflect( rayCone, curvature );
rayCone = scatterBsdf( rayCone, bsdfVal );
```

To sample a texture using the ray cone, first project the ray cone onto the surface based on the normal, and then compute the texture derivatives from the projected ray differentials. The following code does this for a triangle with vertices `(Va,Vb,Vc)` and texture coordinates `(Ta,Tb,Tc)`:

```
float2 dPdx, dPdy; // ray differentials
projectToRayDifferentialsOnSurface( rayCone.width, rayDirection, normal, dPdx, dPdy );

float2 ddx, ddy; // texture derivatives
computeTexGradientsForTriangle( Va, Vb, Vc, Ta, Tb, Tc, dPdx, dPdy, ddx, ddy );
```

Then sample the texture using the texture derivatives just computed:

```
float4 tex = tex2DGrad<float4>( texture, s, t, ddx, ddy );
```

Depending on how the code is structured, these steps might all reside in *raygen*, or be divided between *raygen* and *closesthit* with the ray cone being passed around in the ray payload.