// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/SceneProxy.h"

#include "DemandPbrtScene/GeometryCache.h"
#include "DemandPbrtScene/MaterialAdapters.h"
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

using namespace otk::pbrt;

namespace demandPbrtScene {

inline OptixAabb toOptixAabb( const pbrt::Bounds3f& bounds )
{
    return OptixAabb{ bounds.pMin.x, bounds.pMin.y, bounds.pMin.z, bounds.pMax.x, bounds.pMax.y, bounds.pMax.z };
}

inline GeometryPrimitive primitiveForType( const std::string& type )
{
    if( type == SHAPE_TYPE_TRIANGLE_MESH || type == SHAPE_TYPE_PLY_MESH )
        return GeometryPrimitive::TRIANGLE;
    if( type == SHAPE_TYPE_SPHERE )
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

inline PhongMaterial geometryMaterial( const PlasticMaterial& shapeMaterial )
{
    auto          toFloat3 = []( const pbrt::Point3f& pt ) { return make_float3( pt.x, pt.y, pt.z ); };
    PhongMaterial material{};
    material.Ka    = toFloat3( shapeMaterial.Ka );
    material.Kd    = toFloat3( shapeMaterial.Kd );
    material.Ks    = toFloat3( shapeMaterial.Ks );
    material.flags = plasticMaterialFlags( shapeMaterial );
    return material;
}

inline MaterialGroup materialGroupForMaterial( const PlasticMaterial& material, uint_t primitiveIndexEnd )
{
    return { geometryMaterial( material ),  //
             material.diffuseMapFileName,   //
             material.alphaMapFileName,     //
             primitiveIndexEnd };
}

inline uint_t proxyMaterialSbtOffsetForPrimitive( GeometryPrimitive primitive )
{
    switch( primitive )
    {
        case GeometryPrimitive::TRIANGLE:
            return +HitGroupIndex::PROXY_MATERIAL_TRIANGLE;
        case GeometryPrimitive::SPHERE:
            return +HitGroupIndex::PROXY_MATERIAL_SPHERE;
        case GeometryPrimitive::NONE:
            break;
    }
    throw std::runtime_error( "Unsupported primitive " + std::to_string( +primitive ) );
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
        const ShapeDefinition& shape{ m_scene->freeShapes[m_shapeIndex] };
        return toOptixAabb( shape.transform( shape.bounds ) );
    }
    bool isDecomposable() const override { return false; }

    virtual pbrt::Transform getTransform() const { return {}; }

    GeometryInstance createGeometry( OptixDeviceContext context, CUstream stream ) override;

    std::vector<SceneProxyPtr> decompose( ProxyFactoryPtr ) override { return {}; }

  protected:
    uint_t              m_pageId;
    SceneDescriptionPtr m_scene;
    uint_t              m_shapeIndex;

    GeometryInstance createGeometryFromShape( OptixDeviceContext context, CUstream stream, const ShapeDefinition& shape );

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

    pbrt::Transform getTransform() const override { return m_instance.transform; }

    GeometryInstance createGeometry( OptixDeviceContext context, CUstream stream ) override;

  private:
    ObjectInstanceDefinition& m_instance;
    ShapeDefinition&          m_shape;
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
                            GeometryPrimitive   primitive,
                            MaterialFlags       flags );
    ~InstancePrimitiveProxy() override = default;

    uint_t                     getPageId() const override { return m_pageId; }
    OptixAabb                  getBounds() const override { return m_bounds; }
    bool                       isDecomposable() const override { return false; }
    GeometryInstance           createGeometry( OptixDeviceContext context, CUstream stream ) override;
    std::vector<SceneProxyPtr> decompose( ProxyFactoryPtr proxyFactory ) override;

  private:
    const Options&          m_options;
    GeometryCachePtr        m_geometryCache;
    uint_t                  m_pageId;
    OptixAabb               m_bounds;
    SceneDescriptionPtr     m_scene;
    uint_t                  m_instanceIndex;
    std::string             m_name;
    const ObjectDefinition& m_object;
    GeometryPrimitive       m_primitive;
    MaterialFlags           m_flags;
};

InstancePrimitiveProxy::InstancePrimitiveProxy( const Options&      options,
                                                GeometryCachePtr    geometryCache,
                                                uint_t              id,
                                                const OptixAabb&    bounds,
                                                SceneDescriptionPtr scene,
                                                uint_t              instanceIndex,
                                                GeometryPrimitive   primitive,
                                                MaterialFlags       flags )
    : m_options( options )
    , m_geometryCache( std::move( geometryCache ) )
    , m_pageId( id )
    , m_bounds( bounds )
    , m_scene( std::move( scene ) )
    , m_instanceIndex( instanceIndex )
    , m_name( m_scene->objectInstances[instanceIndex].name )
    , m_object( m_scene->objects[m_name] )
    , m_primitive( primitive )
    , m_flags( flags )
{
}

GeometryInstance InstancePrimitiveProxy::createGeometry( OptixDeviceContext context, CUstream stream )
{
    if( m_options.proxyGranularity == ProxyGranularity::FINE )
    {
        throw std::runtime_error( "InstancePrimitiveProxy::createGeometry called for decomposable proxy" );
    }

    ShapeList  shapes{ m_scene->objectShapes[m_name] };
    const auto split{ std::partition( shapes.begin(), shapes.end(), [this]( const ShapeDefinition& shape ) {
        return primitiveForType( shape.type ) == m_primitive  //
               && shapeMaterialFlags( shape ) == m_flags;
    } ) };
    const GeometryCacheEntry result{
        m_geometryCache->getObject( context, stream, m_scene->objects[m_name], shapes, m_primitive, m_flags ) };
    // create MaterialGroups for each shape in the GAS
    int                        i{};
    std::vector<MaterialGroup> groups;
    for( auto it = shapes.begin(); it != split; ++it )
    {
        const ShapeDefinition& shape{ *it };
        groups.push_back( { materialGroupForMaterial( shape.material, result.primitiveGroupIndices[i++] ) } );
    }
    const uint_t sbtOffset{ proxyMaterialSbtOffsetForPrimitive( m_primitive ) };
    return { result.accelBuffer,  //
             m_primitive,         //
             geometryInstance( m_scene->objectInstances[m_instanceIndex].transform, m_pageId, result.traversable, sbtOffset ),  //
             groups,             //
             result.devNormals,  //
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
    // fine proxy granularity is always one proxy per shape
    const std::vector<ShapeDefinition>& shapeDefinitions{ m_scene->objectShapes[m_name] };
    if( m_options.proxyGranularity == ProxyGranularity::FINE )
    {
        return shapeDefinitions.size() > 1;
    }

    // Decomposable if any of the shapes is different from the first
    const ShapeDefinition&  firstShape{ shapeDefinitions.front() };
    const GeometryPrimitive primitive{ primitiveForType( firstShape.type ) };
    const MaterialFlags     flags{ shapeMaterialFlags( firstShape ) };
    auto                    begin{ shapeDefinitions.cbegin() };
    ++begin;

    // shapes with different primitive types or texture types are decomposable
    return std::any_of( begin, shapeDefinitions.cend(), [&]( const ShapeDefinition& shape ) {
        return primitiveForType( shape.type ) != primitive  //
               || shapeMaterialFlags( shape ) != flags;
    } );
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

    const ShapeList&        shapes{ m_scene->objectShapes[m_name] };
    const ShapeDefinition&  shape{ shapes[0] };
    const GeometryPrimitive primitive{ primitiveForType( shape.type ) };
    const MaterialFlags     flags{ shapeMaterialFlags( shape ) };
    const auto              compareShapes{ [=]( const ShapeDefinition& s ) {
        return primitiveForType( s.type ) != primitive || shapeMaterialFlags( s ) != flags;
    } };
    if( std::any_of( ++shapes.begin(), shapes.end(), compareShapes ) )
    {
        throw std::runtime_error( "Attempt to get geometry for decomposable proxy" );
    }

    const GeometryCacheEntry result{ m_geometryCache->getObject( context, stream, m_scene->objects[m_name], shapes, primitive, flags ) };
    // create MaterialGroups for each shape in the GAS
    int                        i{};
    std::vector<MaterialGroup> groups;
    for( const ShapeDefinition& s : shapes )
    {
        groups.push_back( { materialGroupForMaterial( s.material, result.primitiveGroupIndices[i++] ) } );
    }
    const uint_t sbtOffset{ proxyMaterialSbtOffsetForPrimitive( primitive ) };
    return { result.accelBuffer,  //
             primitive,           //
             geometryInstance( m_scene->objectInstances[m_instanceIndex].transform, m_pageId, result.traversable, sbtOffset ),  //
             groups,             //
             result.devNormals,  //
             result.devUVs };

    //const GeometryCacheEntry result{ m_geometryCache->getObject( context, stream, m_scene->objects[m_name], shapes, primitive, flags ) };
    //// TODO: figure out how to convey multiple materials per GAS
    //const MaterialGroup groups{ materialGroupForMaterial( shape.material, 0U ) };
    //const uint_t        sbtOffset{ proxyMaterialSbtOffsetForPrimitive( primitive ) };
    //return { result.accelBuffer,  //
    //         primitive,           //
    //         geometryInstance( m_scene->objectInstances[m_instanceIndex].transform, m_pageId, result.traversable, sbtOffset ),  //
    //         { groups },         //
    //         result.devNormals,  //
    //         result.devUVs };
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
        ShapeList shapes{ m_scene->objectShapes[m_name] };
        auto      split{ shapes.begin() };
        while( split != shapes.end() )
        {
            const GeometryPrimitive primitive{ primitiveForType( split->type ) };
            const MaterialFlags     flags{ shapeMaterialFlags( *split ) };
            proxies.push_back( proxyFactory->sceneInstancePrimitive( m_scene, m_instanceIndex, primitive, flags ) );
            split = std::partition( split, shapes.end(), [=]( const ShapeDefinition& shape ) {
                return primitiveForType( shape.type ) == primitive  //
                       && shapeMaterialFlags( shape ) == flags;
            } );
        }
    }

    return proxies;
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

GeometryInstance ShapeProxy::createGeometryFromShape( OptixDeviceContext context, CUstream stream, const ShapeDefinition& shape )
{
    const GeometryPrimitive primitive{ primitiveForType( shape.type ) };

    uint_t sbtOffset{};
    switch( primitive )
    {
        case GeometryPrimitive::TRIANGLE:
            sbtOffset = +HitGroupIndex::PROXY_MATERIAL_TRIANGLE;
            break;
        case GeometryPrimitive::SPHERE:
            sbtOffset = +HitGroupIndex::PROXY_MATERIAL_SPHERE;
            break;
        case GeometryPrimitive::NONE:
            throw std::runtime_error( "Unsupported shape type " + shape.type );
    }

    const GeometryCacheEntry entry = m_geometryCache->getShape( context, stream, shape );
    return { entry.accelBuffer,                                                                             //
             primitive,                                                                                     //
             geometryInstance( getTransform() * shape.transform, m_pageId, entry.traversable, sbtOffset ),  //
             { materialGroupForMaterial( shape.material, 0U ) },                                            //
             entry.devNormals,                                                                              //
             entry.devUVs };
}

GeometryInstance ShapeProxy::createGeometry( OptixDeviceContext context, CUstream stream )
{
    const ShapeDefinition& shape = m_scene->freeShapes[m_shapeIndex];

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
    SceneProxyPtr sceneInstancePrimitive( SceneDescriptionPtr scene, uint_t instanceIndex, GeometryPrimitive primitive, MaterialFlags flags ) override;

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
    const ShapeDefinition& shape = scene->freeShapes[shapeIndex];
    const uint_t           id    = m_geometryLoader->add( toOptixAabb( shape.transform( shape.bounds ) ) );
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
    const ObjectInstanceDefinition& instance = scene->objectInstances[instanceIndex];
    const ShapeList&                shapes   = scene->objectShapes[instance.name];
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
    const ObjectInstanceDefinition& instance = scene->objectInstances[instanceIndex];
    const ShapeList&                shapes   = scene->objectShapes[instance.name];
    const ShapeDefinition&          shape    = shapes[shapeIndex];
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

SceneProxyPtr ProxyFactoryImpl::sceneInstancePrimitive( SceneDescriptionPtr scene, uint_t instanceIndex, GeometryPrimitive primitive, MaterialFlags flags )
{
    const ObjectInstanceDefinition& instance{ scene->objectInstances[instanceIndex] };
    const ShapeList&                shapes{ scene->objectShapes[instance.name] };
    pbrt::Bounds3f                  bounds{};
    for( const ShapeDefinition& shape : shapes )
    {
        if( primitiveForType( shape.type ) == primitive  //
            && shapeMaterialFlags( shape ) == flags )
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
    ++m_stats.numInstancePrimitiveProxiesCreated;
    ++m_stats.numGeometryProxiesCreated;
    return std::make_shared<InstancePrimitiveProxy>( m_options, std::move( m_geometryCache ), id, toOptixAabb( bounds ),
                                                     std::move( scene ), instanceIndex, primitive, flags );
}

ProxyFactoryPtr createProxyFactory( const Options& options, GeometryLoaderPtr geometryLoader, GeometryCachePtr geometryCache )
{
    return std::make_shared<ProxyFactoryImpl>( options, std::move( geometryLoader ), std::move( geometryCache ) );
}

}  // namespace demandPbrtScene
