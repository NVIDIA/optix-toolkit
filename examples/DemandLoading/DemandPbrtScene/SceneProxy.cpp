// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/SceneProxy.h"

#include "DemandPbrtScene/GeometryCache.h"
#include "DemandPbrtScene/Options.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/SceneAdapters.h"

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

inline GeometryPrimitive primitiveForType( const std::string& type )
{
    if( type == "trianglemesh" || type == "plymesh" )
        return GeometryPrimitive::TRIANGLE;
    if( type == "sphere" )
        return GeometryPrimitive::SPHERE;
    throw std::runtime_error( "Unknown shape type " + type );
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

    std::vector<SceneProxyPtr> decompose( ProxyFactoryPtr proxyFactory ) override;

  private:
    uint_t              m_pageId;
    SceneDescriptionPtr m_scene;
};

class ShapeProxy : public SceneProxy
{
  public:
    ShapeProxy( GeometryCachePtr geometryCache, uint_t pageId, SceneDescriptionPtr scene, uint_t shapeIndex )
        : m_pageId( pageId )
        , m_scene( std::move( scene ) )
        , m_shapeIndex( shapeIndex )
        , m_geometryCache( std::move( geometryCache ) )
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

    virtual ::pbrt::Transform getTransform() const { return {}; }

    GeometryInstance createGeometry( OptixDeviceContext context, CUstream stream ) override;

    std::vector<SceneProxyPtr> decompose( ProxyFactoryPtr ) override { return {}; }

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

class InstancePrimitiveProxy : public SceneProxy
{
  public:
    InstancePrimitiveProxy( const Options&      options,
                            GeometryCachePtr    geometryCache,
                            uint_t              id,
                            const OptixAabb&    bounds,
                            SceneDescriptionPtr scene,
                            uint_t              instanceIndex,
                            GeometryPrimitive   primitive );
    ~InstancePrimitiveProxy() override = default;

    uint_t                     getPageId() const override { return m_pageId; }
    OptixAabb                  getBounds() const override { return m_bounds; }
    bool                       isDecomposable() const override { return false; }
    GeometryInstance           createGeometry( OptixDeviceContext context, CUstream stream ) override;
    std::vector<SceneProxyPtr> decompose( ProxyFactoryPtr proxyFactory ) override;

  private:
    const Options&                               m_options;
    GeometryCachePtr                             m_geometryCache;
    uint_t                                       m_pageId;
    OptixAabb                                    m_bounds;
    SceneDescriptionPtr                          m_scene;
    uint_t                                       m_instanceIndex;
    std::string                                  m_name;
    const ::otk::pbrt::ObjectDefinition&         m_object;
    GeometryPrimitive                            m_primitive;
};

InstancePrimitiveProxy::InstancePrimitiveProxy( const Options&      options,
                                                GeometryCachePtr    geometryCache,
                                                uint_t              id,
                                                const OptixAabb&    bounds,
                                                SceneDescriptionPtr scene,
                                                uint_t              instanceIndex,
                                                GeometryPrimitive   primitive )
    : m_options( options )
    , m_geometryCache( std::move( geometryCache ) )
    , m_pageId( id )
    , m_bounds( bounds )
    , m_scene( std::move( scene ) )
    , m_instanceIndex( instanceIndex )
    , m_name( m_scene->objectInstances[instanceIndex].name )
    , m_object( m_scene->objects[m_name] )
    , m_primitive( primitive )
{
}

GeometryInstance InstancePrimitiveProxy::createGeometry( OptixDeviceContext context, CUstream stream )
{
    if( m_options.proxyGranularity == ProxyGranularity::FINE )
    {
        throw std::runtime_error( "InstancePrimitiveProxy::createGeometry called for decomposable proxy" );
    }

    const GeometryCacheEntry result{ m_geometryCache->getObject( context, stream, m_scene->objects[m_name],
                                                                 m_scene->objectShapes[m_name], m_primitive ) };
    uint_t                   sbtOffset{};
    return { result.accelBuffer,  //
             m_primitive,         //
             geometryInstance( m_scene->objectInstances[m_instanceIndex].transform, m_pageId, result.traversable, sbtOffset ),  //
             { {} /*geometryMaterial( shape.material, hasUVs )*/,  //
               {} /*shape.material.diffuseMapFileName*/,           //
               {} /*shape.material.alphaMapFileName*/,             //
               0U },                                               //
             result.devNormals,                                    //
             result.devUVs };
}

std::vector<SceneProxyPtr> InstancePrimitiveProxy::decompose( ProxyFactoryPtr proxyFactory )
{
    throw std::runtime_error( "InstancePrimitiveProxy::decompose called" );
}

class InstanceProxy : public SceneProxy
{
  public:
    InstanceProxy( const Options& options, GeometryCachePtr geometryCache, uint_t pageId, SceneDescriptionPtr scene, uint_t instanceIndex )
        : m_options( options )
        , m_geometryCache( std::move( geometryCache ) )
        , m_pageId( pageId )
        , m_scene( scene )
        , m_instanceIndex( instanceIndex )
        , m_name( scene->objectInstances[instanceIndex].name )
    {
    }

    uint_t    getPageId() const override { return m_pageId; }
    OptixAabb getBounds() const override;
    bool      isDecomposable() const override;

    GeometryInstance createGeometry( OptixDeviceContext context, CUstream stream ) override;

    std::vector<SceneProxyPtr> decompose( ProxyFactoryPtr proxyFactory ) override;

  private:
    // Dependencies
    const Options&   m_options;
    GeometryCachePtr m_geometryCache;

    // Proxy data
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
    const std::vector<otk::pbrt::ShapeDefinition>& shapeDefinitions{ m_scene->objectShapes[m_name] };
    if( m_options.proxyGranularity == ProxyGranularity::FINE )
    {
        return shapeDefinitions.size() > 1;
    }

    // Decomposable if any of the shapes is different from the first
    const std::string type{ shapeDefinitions.front().type };
    auto              begin{ shapeDefinitions.cbegin() };
    ++begin;
    return std::any_of( begin, shapeDefinitions.cend(),
                        [&]( const otk::pbrt::ShapeDefinition& definition ) { return definition.type != type; } );
}

static std::string toString( GeometryPrimitive value )
{
    switch( value )
    {
        case GeometryPrimitive::NONE:
            return "NONE";
        case GeometryPrimitive::TRIANGLE:
            return "TRIANGLE";
        case GeometryPrimitive::SPHERE:
            return "SPHERE";
    }
    return "?Unknown (" + std::to_string( +value ) + ")";
}

GeometryInstance InstanceProxy::createGeometry( OptixDeviceContext context, CUstream stream )
{
    if( m_options.proxyGranularity == ProxyGranularity::FINE )
    {
        throw std::runtime_error( "InstanceProxy::createGeometry called for decomposable proxy" );
    }

    const otk::pbrt::ShapeList& shapes{ m_scene->objectShapes[m_name] };
    auto                        it = shapes.begin();
    const GeometryPrimitive     primitive{ primitiveForType( it->type ) };
    const auto                  compareShapeType{
        [=]( const ::otk::pbrt::ShapeDefinition& shape ) { return primitiveForType( shape.type ) != primitive; } };
    if( std::any_of( ++it, shapes.end(), compareShapeType ) )
    {
        throw std::runtime_error( "Attempt to get geometry for decomposable proxy" );
    }

    const GeometryCacheEntry result{ m_geometryCache->getObject( context, stream, m_scene->objects[m_name], shapes, primitive ) };
    // TODO: figure out how to convey multiple materials per GAS
    OptixInstance instance{};
    instance.traversableHandle = result.traversable;
    return { result.accelBuffer,  //
             primitive,           //
             instance /*geometryInstance( getTransform() * shape.transform, m_pageId, entry.traversable, sbtOffset )*/,  //
             { {} /*geometryMaterial( shape.material, hasUVs )*/,  //
               {} /*shape.material.diffuseMapFileName*/,           //
               {} /*shape.material.alphaMapFileName*/,             //
               0U },                                               //
             result.devNormals,                                    //
             result.devUVs };
}

std::vector<SceneProxyPtr> InstanceProxy::decompose( ProxyFactoryPtr proxyFactory )
{
    std::vector<SceneProxyPtr> proxies;
    if( m_options.proxyGranularity == ProxyGranularity::FINE )
    {
        for( uint_t i = 0; i < static_cast<uint_t>( m_scene->objectShapes[m_name].size() ); ++i )
        {
            proxies.push_back( proxyFactory->sceneInstanceShape( m_scene, m_instanceIndex, i ) );
        }
    }
    else
    {
        ::otk::pbrt::ShapeList shapes{ m_scene->objectShapes[m_name] };
        auto                   split{ shapes.begin() };
        while( split != shapes.end() )
        {
            const GeometryPrimitive primitive{ primitiveForType( split->type ) };
            proxies.push_back( proxyFactory->sceneInstancePrimitive( m_scene, m_instanceIndex, primitive ) );
            split = std::partition( split, shapes.end(), [=]( const ::otk::pbrt::ShapeDefinition& shape ) {
                return primitiveForType( shape.type ) == primitive;
            } );
        }
    }

    return proxies;
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
    throw std::runtime_error( "WholeSceneProxy::createGeometry called" );
}

std::vector<SceneProxyPtr> WholeSceneProxy::decompose( ProxyFactoryPtr proxyFactory )
{
    std::vector<SceneProxyPtr> proxies;
    for( uint_t i = 0; i < static_cast<uint_t>( m_scene->objectInstances.size() ); ++i )
    {
        proxies.push_back( proxyFactory->sceneInstance( m_scene, i ) );
    }
    for( uint_t i = 0; i < static_cast<uint_t>( m_scene->freeShapes.size() ); ++i )
    {
        proxies.push_back( proxyFactory->sceneShape( m_scene, i ) );
    }
    return proxies;
}

GeometryInstance ShapeProxy::createGeometryFromShape( OptixDeviceContext context, CUstream stream, const otk::pbrt::ShapeDefinition& shape )
{
    uint_t            sbtOffset;
    GeometryPrimitive primitive;
    bool              hasUVs = false;
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
    return { entry.accelBuffer,                                                                             //
             primitive,                                                                                     //
             geometryInstance( getTransform() * shape.transform, m_pageId, entry.traversable, sbtOffset ),  //
             MaterialGroup{ geometryMaterial( shape.material, hasUVs ),                                     //
                            shape.material.diffuseMapFileName,                                              //
                            shape.material.alphaMapFileName,                                                //
                            0U },                                                                           //
             entry.devNormals,                                                                              //
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
    ProxyFactoryImpl( const Options& options, GeometryLoaderPtr geometryLoader, GeometryCachePtr geometryCache )
        : m_options( options )
        , m_geometryLoader( std::move( geometryLoader ) )
        , m_geometryCache( std::move( geometryCache ) )
    {
    }
    ~ProxyFactoryImpl() override = default;

    SceneProxyPtr scene( SceneDescriptionPtr scene ) override;
    SceneProxyPtr sceneShape( SceneDescriptionPtr scene, uint_t shapeIndex ) override;
    SceneProxyPtr sceneInstance( SceneDescriptionPtr scene, uint_t instanceIndex ) override;
    SceneProxyPtr sceneInstanceShape( SceneDescriptionPtr scene, uint_t instanceIndex, uint_t shapeIndex ) override;
    SceneProxyPtr sceneInstancePrimitive( SceneDescriptionPtr scene, uint_t instanceIndex, GeometryPrimitive primitive ) override;

    ProxyFactoryStatistics getStatistics() const override { return m_stats; }

  private:
    const Options&         m_options;
    GeometryLoaderPtr      m_geometryLoader;
    GeometryCachePtr       m_geometryCache;
    ProxyFactoryStatistics m_stats{};
};

SceneProxyPtr ProxyFactoryImpl::scene( SceneDescriptionPtr scene )
{
    ++m_stats.numSceneProxiesCreated;

    // One free shape, no instances
    if( scene->freeShapes.size() == 1 && scene->objectInstances.empty() )
    {
        return sceneShape( scene, 0 );
    }

    // One instance, no free shapes
    if( scene->freeShapes.empty() && scene->objectInstances.size() == 1 )
    {
        const uint_t instanceIndex = 0;
        return sceneInstance( scene, instanceIndex );
    }

    const uint_t id = m_geometryLoader->add( toOptixAabb( scene->bounds ) );
    ++m_stats.numGeometryProxiesCreated;
    return std::make_shared<WholeSceneProxy>( id, scene );
}

SceneProxyPtr ProxyFactoryImpl::sceneShape( SceneDescriptionPtr scene, uint_t shapeIndex )
{
    const otk::pbrt::ShapeDefinition& shape = scene->freeShapes[shapeIndex];
    const uint_t                      id    = m_geometryLoader->add( toOptixAabb( shape.transform( shape.bounds ) ) );
    if( m_options.verboseSceneDecomposition )
    {
        std::cout << "Added scene shape[" << shapeIndex << "] as proxy id " << id << '\n';
    }
    ++m_stats.numShapeProxiesCreated;
    ++m_stats.numGeometryProxiesCreated;
    return std::make_shared<ShapeProxy>( m_geometryCache, id, scene, shapeIndex );
}

SceneProxyPtr ProxyFactoryImpl::sceneInstance( SceneDescriptionPtr scene, uint_t instanceIndex )
{
    const otk::pbrt::ObjectInstanceDefinition& instance = scene->objectInstances[instanceIndex];
    const otk::pbrt::ShapeList&                shapes   = scene->objectShapes[instance.name];
    // Instance consists of one shape
    if( shapes.size() == 1 )
    {
        const uint_t shapeIndex = 0;
        return sceneInstanceShape( scene, instanceIndex, shapeIndex );
    }

    // TODO: if all shapes are the same primitive type under COARSE granularity, return sceneInstancePrimitive

    const uint_t id = m_geometryLoader->add( toOptixAabb( instance.transform( instance.bounds ) ) );
    if( m_options.verboseSceneDecomposition )
    {
        std::cout << "Added instance " << instance.name << "[" << instanceIndex << "] as proxy id " << id << '\n';
    }
    ++m_stats.numInstanceProxiesCreated;
    ++m_stats.numGeometryProxiesCreated;
    return std::make_shared<InstanceProxy>( m_options, m_geometryCache, id, scene, instanceIndex );
}

SceneProxyPtr ProxyFactoryImpl::sceneInstanceShape( SceneDescriptionPtr scene, uint_t instanceIndex, uint_t shapeIndex )
{
    const otk::pbrt::ObjectInstanceDefinition& instance = scene->objectInstances[instanceIndex];
    const otk::pbrt::ShapeList&                shapes   = scene->objectShapes[instance.name];
    const otk::pbrt::ShapeDefinition&          shape    = shapes[shapeIndex];
    const uint_t id = m_geometryLoader->add( toOptixAabb( instance.transform( shape.transform( shape.bounds ) ) ) );
    if( m_options.verboseSceneDecomposition )
    {
        std::cout << "Added instance " << instance.name << "[" << instanceIndex << "] shape[" << shapeIndex
                  << "] as proxy id " << id << '\n';
    }
    ++m_stats.numInstanceShapeProxiesCreated;
    ++m_stats.numGeometryProxiesCreated;
    return std::make_shared<InstanceShapeProxy>( m_geometryCache, id, scene, instanceIndex, shapeIndex );
}

SceneProxyPtr ProxyFactoryImpl::sceneInstancePrimitive( SceneDescriptionPtr scene, uint_t instanceIndex, GeometryPrimitive primitive )
{
    const otk::pbrt::ObjectInstanceDefinition& instance{ scene->objectInstances[instanceIndex] };
    const otk::pbrt::ShapeList&                shapes{ scene->objectShapes[instance.name] };
    ::pbrt::Bounds3f                           bounds{};
    for( const otk::pbrt::ShapeDefinition& shape : shapes )
    {
        if( primitiveForType( shape.type ) == primitive )
        {
            bounds = Union( bounds, instance.transform( shape.transform( shape.bounds ) ) );
        }
    }
    const uint_t id = m_geometryLoader->add( toOptixAabb( bounds ) );
    if( m_options.verboseSceneDecomposition )
    {
        std::cout << "Added instance " << instance.name << "[" << instanceIndex << "] primitive "
                  << toString( primitive ) << " as proxy id " << id << '\n';
    }
    ++m_stats.numInstanceShapeProxiesCreated;
    ++m_stats.numGeometryProxiesCreated;
    return std::make_shared<InstancePrimitiveProxy>( m_options, std::move( m_geometryCache ), id, toOptixAabb( bounds ),
                                                     std::move( scene ), instanceIndex, primitive );
}

ProxyFactoryPtr createProxyFactory( const Options& options, GeometryLoaderPtr geometryLoader, GeometryCachePtr geometryCache )
{
    return std::make_shared<ProxyFactoryImpl>( options, std::move( geometryLoader ), std::move( geometryCache ) );
}

}  // namespace demandPbrtScene
