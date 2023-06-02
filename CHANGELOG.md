# OptiX Toolkit Shader Util changes

## Version 0.8.1

* Helper functions in [ray_cone.h](include/OptiXToolkit/ShaderUtil/ray_cone.h) facilitate using ray
cones to drive texture filtering.  See the [Ray Cones whitepaper](docs/RayCones.pdf) for more information.
An [example application](https://github.com/NVIDIA/otk-examples/DemandLoading/RayCones) illustrates
how to use ray cones with the [Demand Loading library](https://github.com/NVIDIA/otk-demand-loading).

## Version 0.8

* The self intersection avoidance library computes spawn points for secondary rays, safe from
self-intersections.  See the [README](README.md) for more information.
