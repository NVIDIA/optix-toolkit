# OptiX Demand Loading Library Change Log

## Version 0.8

The demand loading library was factored into two layers:
- A layer that deals only in requested page ids
- A layer that maps requested page ids to textures and resources

This change should not affect clients of the existing demand loading
library.  Applications that wish to employ their own mapping of requested
pages to resources can use the lower layer.

A simple demand load geometry library was added.  Instead of loading geometry
directly into the scene, the application adds proxies associated with a bounding
box.  After launch, the demand geometry library reports back which proxies were
intersected and the application can use this to load the corresponding geometry.
See the [DemandGeometry ReadMe](DemandGeometry/ReadMe.md) for details on the API.
