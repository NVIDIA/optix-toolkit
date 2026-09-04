<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-->

# PBRT Material Support in DemandPbrtScene

`DemandPbrtScene` loads PBRT v3 scenes and preserves the original PBRT material
and texture-graph intent so material support can improve without changing the
scene loader. The renderer still keeps a small local CUDA fallback, but
the primary path for broad PBRT material support is generated MDL compiled
through the MDL SDK.

The implementation is approximate with respect to PBRT. It is intended to render
PBRT material families through physically based GPU BSDF evaluation where that is
practical, while keeping unsupported or still-compiling material features visible
through fallback shading.

## Architecture

Material loading is demand-driven. Geometry initially renders through proxy
materials. When a proxy material page is touched by a ray, `MaterialResolver`
resolves the PBRT material for that durable material ID. If the material is
eligible for generated MDL, the resolver creates or finds a structural
`MdlShaderKey`, binds the instance-specific values and demand textures, and
publishes the resulting material state.

The SBT is used only to select OptiX program-group combinations. SBT records are
header-only `Record<EmptyData>` entries; material payloads are not stored in SBT
records. Device-side material data is uploaded through launch-parameter arrays
indexed by material ID:

- `materialStates` selects the active backend and fallback reason.
- `mdlMaterialShaders` stores MDL callable indices, argument block pointers, and
  fixed runtime texture bindings for MDL-ready materials.
- `fourierMaterialResources` stores PBRT Fourier table resource IDs and device
  table descriptors.
- `realizedMaterials` and `partialMaterials` store local fallback material data.

Closest-hit programs recover the material ID from the instance and primitive
material range data, then index the appropriate launch-parameter array. This
keeps material data flow explicit and separate from SBT slot selection.

## Demand-Compiled MDL

With `--mdl-generated-materials`, supported PBRT materials are converted to
generated MDL source. With `--mdl-material-delay`, compilation and OptiX pipeline
creation happen asynchronously:

1. The first demanded material page reserves a stable, data-free SBT slot that
   initially points at a fallback hit group.
2. A worker thread compiles the generated MDL target code, creates the needed
   OptiX modules and program groups, and links a candidate pipeline.
3. The render thread adopts the completed pipeline at a frame boundary and
   uploads the `MdlMaterialShader` payload through the material-ID-indexed device
   array.
4. Rendering continues with the same stable SBT offset, now selecting the MDL
   closest-hit program instead of fallback.

If compilation is pending, fails, or the material uses unsupported PBRT features,
the material stays on visible fallback shading. Without delay mode, the first
demanded material compiles synchronously.

## Supported Material Families

Generated MDL currently handles these PBRT material types for triangle geometry:

- `matte`
- `plastic`
- `uber`
- `mirror`
- `glass`
- `metal`
- `substrate`
- `translucent`
- `subsurface`
- `kdsubsurface`
- simple two-branch `mix`

The generated MDL path uses MDL BSDF init, sample, evaluate, and PDF callables
for MDL-ready materials. Direct lighting uses MDL BSDF evaluation, and path mode
uses MDL BSDF sampling to provide secondary-ray direction and throughput.

Several mappings are approximations rather than PBRT-exact implementations. For
example, `subsurface` and `kdsubsurface` use cheap diffuse/transmission
approximations rather than full BSSRDF transport, and roughness, conductor, and
bump behavior are practical approximations of PBRT semantics.

## Texture Graphs

Generated MDL texture source can represent:

- `imagemap`
- `constant`
- `scale`
- `mix`
- 2D `checkerboard`

Image maps and CPU-generated checkerboards are emitted as demand-texture
placeholders so the generated source remains structural and can be reused across
instances. Runtime MDL texture bindings live in each `MdlMaterialShader` entry as
a fixed 19-slot table for supported parameters such as `Kd`, `Ks`, `Kr`, `Kt`,
roughness, bump maps, `mix` branch textures, and `mix.amount`.

Unsupported procedural textures and unsupported graph shapes are recorded as
explicit fallback reasons instead of being silently approximated.

## Fourier Materials

PBRT `fourier` materials are handled by a PBRT-specific CUDA backend rather than
generated MDL. PBRT `.bsdf` tables do not map directly to MDL measured-BSDF
resources, so the loader resolves and validates PBRT Fourier tables, uploads
them as device resources, and publishes `FourierMaterialResource` entries indexed
by material ID. The Fourier SBT slot selects the Fourier closest-hit program; the
table resource data is read from the launch-parameter array.

## Build and Test Coverage

MDL material support is build-time optional through `OTK_USE_MDL`. MDL reference
tests and generated-material comparisons are registered only when that option is
enabled. Non-MDL builds keep the local fallback path available for approximate
rendering, unsupported features, and pending or failed generated MDL compiles.

Material support is validated with focused `DemandPbrtScene` image comparisons
against PBRT reference renders where practical, plus unit coverage for material
resolution, generated source, runtime bindings, and fallback behavior.
