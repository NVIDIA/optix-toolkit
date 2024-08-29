// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

// optix.h uses std::min/std::max
#ifdef WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "DemandPbrtScene/Dependencies.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/Primitive.h"
#include "DemandPbrtScene/ProxyFactoryStatistics.h"

#include <optix.h>

#include <string>
#include <vector>

namespace demandPbrtScene {

struct MaterialGroup
{
    PhongMaterial material;
    std::string   diffuseMapFileName;
    std::string   alphaMapFileName;
    uint_t        primitiveIndexEnd;
};

struct GeometryInstance
{
    CUdeviceptr                accelBuffer;
    GeometryPrimitive          primitive;
    OptixInstance              instance;
    std::vector<MaterialGroup> groups;      // {Material,index} for each group
    TriangleNormals*           devNormals;  // device pointer to per-primitive normals, nullptr if none
    TriangleUVs*               devUVs;      // device pointer to per-vertex UVs, nullptr if none
};

class SceneProxy
{
  public:
    virtual ~SceneProxy() = default;

    /// Returns the proxy geometry id associated with this proxy.
    virtual uint_t getPageId() const = 0;

    /// Returns the world space bounds of the proxy.
    virtual OptixAabb getBounds() const = 0;

    /// Returns true if this proxy can be decomposed into further proxies.
    virtual bool isDecomposable() const = 0;

    /// For a non-decomposable proxy, creates an acceleration structure for the geometry and returns a GeometryInstance.
    virtual GeometryInstance createGeometry( OptixDeviceContext context, CUstream stream ) = 0;

    /// Decomposes a proxy into an array of other proxies.
    virtual std::vector<SceneProxyPtr> decompose( ProxyFactoryPtr proxyFactory ) = 0;
};

class ProxyFactory
{
  public:
    virtual ~ProxyFactory() = default;

    virtual SceneProxyPtr scene( SceneDescriptionPtr scene ) = 0;

    virtual SceneProxyPtr sceneShape( SceneDescriptionPtr scene, uint_t shapeIndex ) = 0;

    virtual SceneProxyPtr sceneInstance( SceneDescriptionPtr scene, uint_t instanceIndex ) = 0;

    virtual SceneProxyPtr sceneInstanceShape( SceneDescriptionPtr scene, uint_t instanceIndex, uint_t shapeIndex ) = 0;

    virtual SceneProxyPtr sceneInstancePrimitive( SceneDescriptionPtr scene,
                                                  uint_t              instanceIndex,
                                                  GeometryPrimitive   primitive,
                                                  MaterialFlags       flags ) = 0;

    virtual ProxyFactoryStatistics getStatistics() const = 0;
};

ProxyFactoryPtr createProxyFactory( const Options& options, GeometryLoaderPtr geometryLoader, GeometryCachePtr geometryCache );

}  // namespace demandPbrtScene
