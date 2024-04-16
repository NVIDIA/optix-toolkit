//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

// optix.h uses std::min/std::max
#ifdef WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "Dependencies.h"
#include "Params.h"
#include "Primitive.h"
#include "ProxyFactoryStatistics.h"

#include <optix.h>

#include <string>
#include <vector>

namespace demandPbrtScene {

struct GeometryInstance
{
    CUdeviceptr       accelBuffer;
    GeometryPrimitive primitive;
    OptixInstance     instance;
    PhongMaterial     material;
    std::string       diffuseMapFileName;
    std::string       alphaMapFileName;
    TriangleNormals*  normals;  // nullptr if no per-vertex normals
    TriangleUVs*      uvs;      // nullptr if no per-vertex UVs
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
    virtual std::vector<SceneProxyPtr> decompose( GeometryLoaderPtr geometryLoader, ProxyFactoryPtr proxyFactory ) = 0;
};

class ProxyFactory
{
  public:
    virtual ~ProxyFactory() = default;

    virtual SceneProxyPtr scene( GeometryLoaderPtr geometryLoader, SceneDescriptionPtr scene ) = 0;
    virtual SceneProxyPtr sceneShape( GeometryLoaderPtr geometryLoader, SceneDescriptionPtr scene, uint_t shapeIndex ) = 0;
    virtual SceneProxyPtr sceneInstance( GeometryLoaderPtr geometryLoader, SceneDescriptionPtr scene, uint_t instanceIndex ) = 0;
    virtual SceneProxyPtr sceneInstanceShape( GeometryLoaderPtr   geometryLoader,
                                              SceneDescriptionPtr scene,
                                              uint_t              instanceIndex,
                                              uint_t              shapeIndex ) = 0;

    virtual ProxyFactoryStatistics getStatistics() const = 0;
};

ProxyFactoryPtr createProxyFactory( const Options& options, GeometryCachePtr geometryCache );

}  // namespace demandPbrtScene
