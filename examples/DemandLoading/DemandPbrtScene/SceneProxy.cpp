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

#include "SceneProxy.h"

#include "GeometryCache.h"
#include "Options.h"
#include "Params.h"
#include "SceneAdapters.h"

#include <OptiXToolkit/DemandGeometry/GeometryLoader.h>
#include <OptiXToolkit/Memory/SyncVector.h>
#include <OptiXToolkit/PbrtSceneLoader/MeshReader.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <optix_stubs.h>

#include <vector_types.h>

#include <algorithm>

namespace demandPbrtScene {

inline OptixAabb toOptixAabb( const ::pbrt::Bounds3f& bounds )
{
    return OptixAabb{ bounds.pMin.x, bounds.pMin.y, bounds.pMin.z, bounds.pMax.x, bounds.pMax.y, bounds.pMax.z };
}

class WholeSceneProxy : public SceneProxy
{
  public:
    WholeSceneProxy( uint_t pageId, SceneDescriptionPtr scene )
        : m_pageId( pageId )
        , m_scene( scene )
    {
    }
    ~WholeSceneProxy() override = default;

    uint_t    getPageId() const override { return m_pageId; }
    OptixAabb getBounds() const override { return toOptixAabb( m_scene->bounds ); }
    bool      isDecomposable() const override { return true; }

    GeometryInstance createGeometry( OptixDeviceContext context, CUstream stream ) override;

    std::vector<SceneProxyPtr> decompose( GeometryLoaderPtr geometryLoader, ProxyFactoryPtr proxyFactory ) override;

  private:
    uint_t              m_pageId;
    SceneDescriptionPtr m_scene;
};

class ShapeProxy : public SceneProxy
{
  public:
    ShapeProxy( GeometryCachePtr geometryCache, uint_t pageId, SceneDescriptionPtr scene, uint_t shapeIndex )
        : m_pageId( pageId )
        , m_scene( scene )
        , m_shapeIndex( shapeIndex )
        , m_geometryCache( geometryCache )
    {
    }
    ~ShapeProxy() override = default;

    uint_t    getPageId() const override { return m_pageId; }
    OptixAabb getBounds() const override
    {
        const otk::pbrt::ShapeDefinition& shape = m_scene->freeShapes[m_shapeIndex];
        return toOptixAabb( shape.transform( shape.bounds ) );
    }
    bool isDecomposable() const override { return false; }

    virtual ::pbrt::Transform getTransform() const { return ::pbrt::Transform(); }

    GeometryInstance createGeometry( OptixDeviceContext context, CUstream stream ) override;

    std::vector<SceneProxyPtr> decompose( GeometryLoaderPtr geometryLoader, ProxyFactoryPtr proxyFactory ) override
    {
        return {};
    }

  protected:
    uint_t              m_pageId;
    SceneDescriptionPtr m_scene;
    uint_t              m_shapeIndex;

    GeometryInstance createGeometryFromShape( OptixDeviceContext context, CUstream stream, const otk::pbrt::ShapeDefinition& shape );

  private:
    GeometryCachePtr               m_geometryCache;
    otk::SyncVector<float3>        m_vertices;
    otk::SyncVector<std::uint16_t> m_indices;
};

class InstanceShapeProxy : public ShapeProxy
{
  public:
    InstanceShapeProxy( GeometryCachePtr geometryCache, uint_t pageId, SceneDescriptionPtr scene, uint_t instanceIndex, uint_t shapeIndex )
        : ShapeProxy( geometryCache, pageId, scene, shapeIndex )
        , m_instance( m_scene->objectInstances[instanceIndex] )
        , m_shape( m_scene->objectShapes[m_instance.name][shapeIndex] )
    {
    }
    ~InstanceShapeProxy() override = default;

    OptixAabb getBounds() const override;

    ::pbrt::Transform getTransform() const override { return m_instance.transform; }

    GeometryInstance createGeometry( OptixDeviceContext context, CUstream stream ) override;

  private:
    otk::pbrt::ObjectInstanceDefinition& m_instance;
    otk::pbrt::ShapeDefinition&          m_shape;
};

OptixAabb InstanceShapeProxy::getBounds() const
{
    return toOptixAabb( m_instance.transform( m_shape.transform( m_shape.bounds ) ) );
}

GeometryInstance InstanceShapeProxy::createGeometry( OptixDeviceContext context, CUstream stream )
{
    return createGeometryFromShape( context, stream, m_shape );
}

class InstanceProxy : public SceneProxy
{
  public:
    InstanceProxy( uint_t pageId, SceneDescriptionPtr scene, uint_t instanceIndex )
        : m_pageId( pageId )
        , m_scene( scene )
        , m_instanceIndex( instanceIndex )
        , m_name( scene->objectInstances[instanceIndex].name )
    {
    }

    uint_t    getPageId() const override { return m_pageId; }
    OptixAabb getBounds() const override;
    bool      isDecomposable() const override;

    GeometryInstance createGeometry( OptixDeviceContext context, CUstream stream ) override;

    std::vector<SceneProxyPtr> decompose( GeometryLoaderPtr geometryLoader, ProxyFactoryPtr proxyFactory ) override;

  private:
    uint_t              m_pageId;
    SceneDescriptionPtr m_scene;
    uint_t              m_instanceIndex;
    std::string         m_name;
};

OptixAabb InstanceProxy::getBounds() const
{
    const auto& instance = m_scene->objectInstances[m_instanceIndex];
    return toOptixAabb( instance.transform( instance.bounds ) );
}

bool InstanceProxy::isDecomposable() const
{
    return m_scene->objectShapes[m_name].size() > 1;
}

GeometryInstance InstanceProxy::createGeometry( OptixDeviceContext context, CUstream stream )
{
    return {};
}

std::vector<SceneProxyPtr> InstanceProxy::decompose( GeometryLoaderPtr geometryLoader, ProxyFactoryPtr proxyFactory )
{
    std::vector<SceneProxyPtr> proxies;
    for( uint_t i = 0; i < static_cast<uint_t>( m_scene->objectShapes[m_name].size() ); ++i )
    {
        proxies.push_back( proxyFactory->sceneInstanceShape( geometryLoader, m_scene, m_instanceIndex, i ) );
    }

    return proxies;
}

inline OptixInstance geometryInstance( const pbrt::Transform& transform, uint_t pageId, OptixTraversableHandle traversable, uint_t sbtOffset )
{
    OptixInstance instance{};
    toInstanceTransform( instance.transform, transform );
    instance.instanceId        = pageId;
    instance.visibilityMask    = 255U;
    instance.traversableHandle = traversable;
    instance.sbtOffset         = sbtOffset;
    return instance;
}

inline PhongMaterial geometryMaterial( const ::otk::pbrt::PlasticMaterial& shapeMaterial, bool hasUVs )
{
    auto          toFloat3 = []( const ::pbrt::Point3f& pt ) { return make_float3( pt.x, pt.y, pt.z ); };
    PhongMaterial material{};
    material.Ka = toFloat3( shapeMaterial.Ka );
    material.Kd = toFloat3( shapeMaterial.Kd );
    material.Ks = toFloat3( shapeMaterial.Ks );
    if( !shapeMaterial.alphaMapFileName.empty() && hasUVs )
    {
        material.flags |= MaterialFlags::ALPHA_MAP;
    }
    if( !shapeMaterial.diffuseMapFileName.empty() && hasUVs )
    {
        material.flags |= MaterialFlags::DIFFUSE_MAP;
    }
    return material;
}

GeometryInstance WholeSceneProxy::createGeometry( OptixDeviceContext context, CUstream stream )
{
    return {};
}

std::vector<SceneProxyPtr> WholeSceneProxy::decompose( GeometryLoaderPtr geometryLoader, ProxyFactoryPtr proxyFactory )
{
    std::vector<SceneProxyPtr> proxies;
    for( uint_t i = 0; i < static_cast<uint_t>( m_scene->objectInstances.size() ); ++i )
    {
        proxies.push_back( proxyFactory->sceneInstance( geometryLoader, m_scene, i ) );
    }
    for( uint_t i = 0; i < static_cast<uint_t>( m_scene->freeShapes.size() ); ++i )
    {
        proxies.push_back( proxyFactory->sceneShape( geometryLoader, m_scene, i ) );
    }
    return proxies;
}

GeometryInstance ShapeProxy::createGeometryFromShape( OptixDeviceContext context, CUstream stream, const otk::pbrt::ShapeDefinition& shape )
{
    uint_t            sbtOffset;
    GeometryPrimitive primitive;
    bool hasUVs = false;
    if( shape.type == "trianglemesh" || shape.type == "plymesh" )
    {
        sbtOffset = +HitGroupIndex::PROXY_MATERIAL_TRIANGLE;
        primitive = GeometryPrimitive::TRIANGLE;
        hasUVs    = shape.type == "trianglemesh" ? !shape.triangleMesh.uvs.empty() :
                                                   shape.plyMesh.loader->getMeshInfo().numTextureCoordinates > 0;
    }
    else if( shape.type == "sphere" )
    {
        sbtOffset = +HitGroupIndex::PROXY_MATERIAL_SPHERE;
        primitive = GeometryPrimitive::SPHERE;
    }
    else
    {
        throw std::runtime_error( "Unsupported shape type " + shape.type );
    }

    GeometryCacheEntry entry = m_geometryCache->getShape( context, stream, shape );
    return { entry.accelBuffer,
             primitive,
             geometryInstance( getTransform() * shape.transform, m_pageId, entry.traversable, sbtOffset ),
             geometryMaterial( shape.material, hasUVs ),
             shape.material.diffuseMapFileName,
             shape.material.alphaMapFileName,
             entry.devNormals,
             entry.devUVs };
}

GeometryInstance ShapeProxy::createGeometry( OptixDeviceContext context, CUstream stream )
{
    const ::otk::pbrt::ShapeDefinition& shape = m_scene->freeShapes[m_shapeIndex];

    return createGeometryFromShape( context, stream, shape );
}

class ProxyFactoryImpl : public ProxyFactory
{
  public:
    ProxyFactoryImpl( const Options& options, GeometryCachePtr geometryCache )
        : m_options( options )
        , m_geometryCache( std::move( geometryCache ) )
    {
    }
    ~ProxyFactoryImpl() override = default;

    SceneProxyPtr scene( GeometryLoaderPtr geometryLoader, SceneDescriptionPtr scene ) override;
    SceneProxyPtr sceneShape( GeometryLoaderPtr geometryLoader, SceneDescriptionPtr scene, uint_t shapeIndex ) override;
    SceneProxyPtr sceneInstance( GeometryLoaderPtr geometryLoader, SceneDescriptionPtr scene, uint_t instanceIndex ) override;
    SceneProxyPtr sceneInstanceShape( GeometryLoaderPtr geometryLoader, SceneDescriptionPtr scene, uint_t instanceIndex, uint_t shapeIndex ) override;

    ProxyFactoryStatistics getStatistics() const override { return m_stats; }

  private:
    const Options&         m_options;
    GeometryCachePtr       m_geometryCache;
    ProxyFactoryStatistics m_stats{};
};

SceneProxyPtr ProxyFactoryImpl::scene( GeometryLoaderPtr geometryLoader, SceneDescriptionPtr scene )
{
    // One free shape, no instances
    if( scene->freeShapes.size() == 1 && scene->objectInstances.empty() )
    {
        return sceneShape( geometryLoader, scene, 0 );
    }

    // One instance, no free shapes
    if( scene->freeShapes.empty() && scene->objectInstances.size() == 1 )
    {
        const uint_t instanceIndex = 0;
        return sceneInstance( geometryLoader, scene, instanceIndex );
    }

    const uint_t id = geometryLoader->add( toOptixAabb( scene->bounds ) );
    ++m_stats.numGeometryProxiesCreated;
    return std::make_shared<WholeSceneProxy>( id, scene );
}

SceneProxyPtr ProxyFactoryImpl::sceneShape( GeometryLoaderPtr geometryLoader, SceneDescriptionPtr scene, uint_t shapeIndex )
{
    const otk::pbrt::ShapeDefinition& shape = scene->freeShapes[shapeIndex];
    const uint_t                      id    = geometryLoader->add( toOptixAabb( shape.transform( shape.bounds ) ) );
    if( m_options.verboseSceneDecomposition )
    {
        std::cout << "Added scene shape[" << shapeIndex << "] as proxy id " << id << '\n';
    }
    ++m_stats.numGeometryProxiesCreated;
    return std::make_shared<ShapeProxy>( m_geometryCache, id, scene, shapeIndex );
}

SceneProxyPtr ProxyFactoryImpl::sceneInstance( GeometryLoaderPtr geometryLoader, SceneDescriptionPtr scene, uint_t instanceIndex )
{
    const otk::pbrt::ObjectInstanceDefinition& instance = scene->objectInstances[instanceIndex];
    const otk::pbrt::ShapeList&                shapes   = scene->objectShapes[instance.name];
    // Instance consists of one shape
    if( shapes.size() == 1 )
    {
        const uint_t shapeIndex = 0;
        return sceneInstanceShape( geometryLoader, scene, instanceIndex, shapeIndex );
    }

    const uint_t id = geometryLoader->add( toOptixAabb( instance.transform( instance.bounds ) ) );
    if( m_options.verboseSceneDecomposition )
    {
        std::cout << "Added instance " << instance.name << "[" << instanceIndex << "] as proxy id " << id << '\n';
    }
    ++m_stats.numGeometryProxiesCreated;
    return std::make_shared<InstanceProxy>( id, scene, instanceIndex );
}

SceneProxyPtr ProxyFactoryImpl::sceneInstanceShape( GeometryLoaderPtr geometryLoader, SceneDescriptionPtr scene, uint_t instanceIndex, uint_t shapeIndex )
{
    const otk::pbrt::ObjectInstanceDefinition& instance = scene->objectInstances[instanceIndex];
    const otk::pbrt::ShapeList&                shapes   = scene->objectShapes[instance.name];
    const otk::pbrt::ShapeDefinition&          shape    = shapes[shapeIndex];
    const uint_t id = geometryLoader->add( toOptixAabb( instance.transform( shape.transform( shape.bounds ) ) ) );
    if( m_options.verboseSceneDecomposition )
    {
        std::cout << "Added instance " << instance.name << "[" << instanceIndex << "] shape[" << shapeIndex
                  << "] as proxy id " << id << '\n';
    }
    ++m_stats.numGeometryProxiesCreated;
    return std::make_shared<InstanceShapeProxy>( m_geometryCache, id, scene, instanceIndex, shapeIndex );
}

ProxyFactoryPtr createProxyFactory( const Options& options, GeometryCachePtr geometryCache )
{
    return std::make_shared<ProxyFactoryImpl>( options, std::move( geometryCache ) );
}

}  // namespace demandPbrtScene
