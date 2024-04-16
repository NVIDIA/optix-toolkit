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

#include <OptiXToolkit/PbrtSceneLoader/PbrtApiImpl.h>

#include <OptiXToolkit/PbrtApi/PbrtApi.h>
#include <OptiXToolkit/PbrtSceneLoader/Logger.h>
#include <OptiXToolkit/PbrtSceneLoader/MeshReader.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <core/api.h>
#include <core/geometry.h>
#include <core/paramset.h>
#include <core/transform.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Follow the semantics as described at https://pbrt.org/fileformat-v3

namespace otk {
namespace pbrt {

class PbrtApiImpl : public Api
{
public:
    PbrtApiImpl( const char* programName, std::shared_ptr<Logger> logger, std::shared_ptr<MeshInfoReader> infoReader );
    ~PbrtApiImpl() override;

    SceneDescriptionPtr parseFile( std::string filename ) override;
    SceneDescriptionPtr parseString( std::string str ) override;

    void identity() override;
    void translate( float dx, float dy, float dz ) override;
    void rotate( float angle, float ax, float ay, float az ) override;
    void scale( float sx, float sy, float sz ) override;
    void lookAt( float ex, float ey, float ez, float lx, float ly, float lz, float ux, float uy, float uz ) override;
    void concatTransform( float transform[16] ) override;
    void transform( float transform[16] ) override;
    void coordinateSystem( const std::string& name ) override;
    void coordSysTransform( const std::string& name ) override;
    void activeTransformAll() override;
    void activeTransformEndTime() override;
    void activeTransformStartTime() override;
    void transformTimes( float start, float end ) override;
    void pixelFilter( const std::string& name, const ParamSet& params ) override;
    void film( const std::string& type, const ParamSet& params ) override;
    void sampler( const std::string& name, const ParamSet& params ) override;
    void accelerator( const std::string& name, const ParamSet& params ) override;
    void integrator( const std::string& name, const ParamSet& params ) override;
    void camera( const std::string& name, const ParamSet& cameraParams ) override;
    void makeNamedMedium( const std::string& name, const ParamSet& params ) override;
    void mediumInterface( const std::string& insideName, const std::string& outsideName ) override;
    void worldBegin() override;
    void attributeBegin() override;
    void attributeEnd() override;
    void transformBegin() override;
    void transformEnd() override;
    void texture( const std::string& name, const std::string& type, const std::string& tex_type, const ParamSet& params ) override;
    void material( const std::string& name, const ParamSet& params ) override;
    void makeNamedMaterial( const std::string& name, const ParamSet& params ) override;
    void namedMaterial( const std::string& name ) override;
    void lightSource( const std::string& name, const ParamSet& params ) override;
    void areaLightSource( const std::string& name, const ParamSet& params ) override;
    void shape( const std::string& type, const ParamSet& params ) override;
    void reverseOrientation() override;
    void objectBegin( const std::string& name ) override;
    void objectEnd() override;
    void objectInstance( const std::string& name ) override;
    void worldEnd() override;

    void info( std::string text, const char* file, int line ) const override;
    void warning( std::string text, const char* file, int line ) const override;
    void error( std::string text, const char* file, int line ) const override;

private:
    // Notion of a pbrt texture: a texture type and a bag of parameters
    struct TextureDefinition
    {
        std::string      type;
        ::pbrt::ParamSet params;
    };

    // Notion of a pbrt material: a material type name ("matte", "uber", etc.) and a bag of parameters
    struct MaterialDefinition
    {
        std::string      name;
        ::pbrt::ParamSet params;
    };

    // Current graphics state maintained while parsing the scene
    struct GraphicsState
    {
        MaterialDefinition currentMaterial;  // pbrt predefined material type names: "uber", "matte", "plastic", etc.
        std::map<std::string, MaterialDefinition> namedMaterials;
        std::map<std::string, TextureDefinition>  floatTextures;
        std::map<std::string, TextureDefinition>  spectrumTextures;
        std::string                               currentNamedMaterial;
        bool                                      reverseOrientation{};
    };

    bool        findParam( const std::string& name, const GraphicsState& state, const MaterialDefinition& material, ::pbrt::Point3f& result ) const;
    bool        findParam( const std::string& name, const GraphicsState& state, ::pbrt::Point3f& result ) const;
    std::string findTexture( const std::string& name, const GraphicsState& state, const MaterialDefinition&material ) const;
    std::string findTexture( const std::string& name, const GraphicsState& state ) const;

    void resetState();
    bool currentObjectMatchesName() const
    {
        // empty object name always matches
        return m_objectName.empty() || m_objectName == m_currentObjectName;
    }
    void            requireInWorld( const char* name );
    bool            insideObject() const { return !m_currentObjectName.empty(); }
    void            addShape( ShapeDefinition shape );
    ShapeDefinition createPlyMesh( const ::pbrt::ParamSet& params );
    ShapeDefinition createTriangleMesh( const ::pbrt::ParamSet& params );
    ShapeDefinition createSphere( const ::pbrt::ParamSet& params );
    PlasticMaterial getShapeMaterial( const ::pbrt::ParamSet& params ) const;
    ::pbrt::Point3f lookupParam( const std::string& name, const ::pbrt::ParamSet& params, ::pbrt::Point3f def ) const;
    std::string     lookupTextureName( const std::string& name, const ::pbrt::ParamSet& params ) const;
    std::string     lookupSpectrumTextureFileName( const std::string& name, const ::pbrt::ParamSet& params ) const;
    std::string     lookupFloatTextureFileName( const std::string& name, const ::pbrt::ParamSet& params ) const;

    void dropEmptyObjects();

private:
    // Dependencies
    std::shared_ptr<Logger>         m_logger;
    std::shared_ptr<MeshInfoReader> m_infoReader;

    // State while parsing
    ::pbrt::Transform                        m_currentTransform;
    std::map<std::string, ::pbrt::Transform> m_coordinateSystems;
    std::vector<::pbrt::Transform>           m_transformStack;
    GraphicsState                            m_graphicsState;
    std::vector<GraphicsState>               m_graphicsStateStack;
    std::string                              m_objectName;
    std::string                              m_currentObjectName;
    ::pbrt::Bounds3f                         m_currentBounds;
    std::vector<::pbrt::Bounds3f>            m_boundsStack;
    bool                                     m_inWorld{};

    // Result of parse.
    SceneDescriptionPtr m_scene;
};

PbrtApiImpl::PbrtApiImpl( const char* programName, std::shared_ptr<Logger> logger, std::shared_ptr<MeshInfoReader> infoReader )
    : m_logger( std::move( logger ) )
    , m_infoReader( std::move( infoReader ) )
{
    m_logger->start( programName );
    setApi( this );
    resetState();
}

PbrtApiImpl::~PbrtApiImpl()
{
    m_logger->stop();
    setApi( nullptr );
}

// The parseXXX methods decouple us from the top-level parsing functions.
SceneDescriptionPtr PbrtApiImpl::parseFile( std::string filename )
{
    resetState();
    ::pbrt::pbrtParseFile( std::move( filename ) );
    dropEmptyObjects();
    return m_scene;
}

SceneDescriptionPtr PbrtApiImpl::parseString( std::string str )
{
    resetState();
    ::pbrt::pbrtParseString( std::move( str ) );
    dropEmptyObjects();
    return m_scene;
}

void PbrtApiImpl::dropEmptyObjects()
{
    std::vector<std::string> emptyObjects;

    for( const std::pair<const std::string, ShapeList>& objectDef : m_scene->objectShapes )
    {
        if( objectDef.second.empty() )
            emptyObjects.push_back( objectDef.first );
    }

    for( const std::string& name : emptyObjects )
    {
        m_scene->objects.erase( name );
        m_scene->objectShapes.erase( name );
        m_scene->instanceCounts.erase( name );
        m_scene->objectInstances.erase(
            std::remove_if( m_scene->objectInstances.begin(), m_scene->objectInstances.end(),
                            [&]( const ObjectInstanceDefinition& instance ) { return instance.name == name; } ),
            m_scene->objectInstances.end() );
    }
}

std::shared_ptr<Api> createApi( const char* programName, const std::shared_ptr<Logger>& logger, const std::shared_ptr<MeshInfoReader>& infoReader )
{
    return std::make_shared<PbrtApiImpl>( programName, logger, infoReader );
}

void PbrtApiImpl::identity()
{
    m_currentTransform = ::pbrt::Transform();
}

void PbrtApiImpl::translate( float dx, float dy, float dz )
{
    m_currentTransform = m_currentTransform * Translate( ::pbrt::Vector3f( dx, dy, dz ) );
}

void PbrtApiImpl::rotate( float angle, float ax, float ay, float az )
{
    m_currentTransform = m_currentTransform * Rotate( angle, ::pbrt::Vector3f( ax, ay, az ) );
}

void PbrtApiImpl::scale( float sx, float sy, float sz )
{
    m_currentTransform = m_currentTransform * ::pbrt::Scale( sx, sy, sz );
}

void PbrtApiImpl::lookAt( float ex, float ey, float ez, float lx, float ly, float lz, float ux, float uy, float uz )
{
    m_scene->lookAt.eye    = ::pbrt::Point3f( ex, ey, ez );
    m_scene->lookAt.lookAt = ::pbrt::Point3f( lx, ly, lz );
    m_scene->lookAt.up     = ::pbrt::Vector3f( ux, uy, uz );
    m_currentTransform = m_currentTransform * LookAt( m_scene->lookAt.eye, m_scene->lookAt.lookAt, m_scene->lookAt.up );
}

void PbrtApiImpl::concatTransform( float transform[16] )
{
    m_currentTransform = m_currentTransform
                         * ::pbrt::Transform( ::pbrt::Matrix4x4( transform[0], transform[4], transform[8], transform[12],
                                                                 transform[1], transform[5], transform[9], transform[13],
                                                                 transform[2], transform[6], transform[10], transform[14],
                                                                 transform[3], transform[7], transform[11], transform[15] ) );
}

void PbrtApiImpl::transform( float transform[16] )
{
    m_currentTransform = ::pbrt::Transform( ::pbrt::Matrix4x4( transform[0], transform[4], transform[8], transform[12],  //
                                                               transform[1], transform[5], transform[9], transform[13],  //
                                                               transform[2], transform[6], transform[10], transform[14],  //
                                                               transform[3], transform[7], transform[11], transform[15] ) );  //
}

void PbrtApiImpl::coordinateSystem( const std::string& name )
{
    m_coordinateSystems[name] = m_currentTransform;
}

void PbrtApiImpl::info( std::string text, const char *file, int line ) const
{
    m_logger->info( std::move( text ), file, line );
}

void PbrtApiImpl::warning( std::string text, const char *file, int line ) const
{
    ++m_scene->warnings;
    m_logger->warning( std::move( text ), file, line );
}

void PbrtApiImpl::error( std::string text, const char *file, int line ) const
{
    ++m_scene->errors;
    m_logger->error( std::move( text ), file, line );
}

#define PBRT_INFO( text_ ) info( text_, __FILE__, __LINE__ )
#define PBRT_WARNING( text_ ) warning( text_, __FILE__, __LINE__ )
#define PBRT_ERROR( text_ ) error( text_, __FILE__, __LINE__ )

void PbrtApiImpl::coordSysTransform( const std::string& name )
{
    const auto it = m_coordinateSystems.find( name );
    if( it != m_coordinateSystems.end() )
    {
        m_currentTransform = it->second;
    }
    else
    {
        PBRT_ERROR( "Unknown coordinate system '" + name + "'" );
    }
}

void PbrtApiImpl::activeTransformAll()
{
    PBRT_WARNING( "ActiveTransform All not implemented." );
}

void PbrtApiImpl::activeTransformEndTime()
{
    PBRT_WARNING( "ActiveTransform EndTime not implemented." );
}

void PbrtApiImpl::activeTransformStartTime()
{
    PBRT_WARNING( "ActiveTransform StartTime not implemented." );
}

void PbrtApiImpl::transformTimes( float start, float end )
{
    PBRT_WARNING( "TransformTimes not implemented." );
}

void PbrtApiImpl::pixelFilter( const std::string& name, const ParamSet& params )
{
    PBRT_WARNING( "PixelFilter '" + name + "' not implemented." );
}

void PbrtApiImpl::film( const std::string& type, const ParamSet& params )
{
    PBRT_WARNING( "Film '" + type + "' not implemented." );
}

void PbrtApiImpl::sampler( const std::string& name, const ParamSet& params )
{
    PBRT_WARNING( "Sampler '" + name + "' not implemented." );
}

void PbrtApiImpl::accelerator( const std::string& name, const ParamSet& params )
{
    PBRT_WARNING( "Accelerator '" + name + "' not implemented." );
}

void PbrtApiImpl::integrator( const std::string& name, const ParamSet& params )
{
    PBRT_WARNING( "Integrator '" + name + "' not implemented." );
}

void PbrtApiImpl::camera( const std::string& name, const ParamSet& cameraParams )
{
    if( name == "perspective" )
    {
        // TODO: use current transformation matrix to identify lookAt, etc.
        int          numVals{};
        const float* fovVal           = cameraParams.FindFloat( "fov", &numVals );
        const float* halfFOVVal       = cameraParams.FindFloat( "halffov", &numVals );
        const float* focalDistanceVal = cameraParams.FindFloat( "focaldistance", &numVals );
        const float* lensRadiusVal    = cameraParams.FindFloat( "lensradius", &numVals );

        const float fov                = halfFOVVal ? 2.0f * *halfFOVVal : fovVal ? *fovVal : 90.0f;
        m_scene->camera.fov            = fov;
        m_scene->camera.focalDistance  = focalDistanceVal ? *focalDistanceVal : 1.0e30f;
        m_scene->camera.lensRadius     = lensRadiusVal ? *lensRadiusVal : 0.0f;
        m_scene->camera.cameraToWorld  = m_currentTransform;
        m_scene->camera.cameraToScreen = ::pbrt::Perspective( fov, 1e-2f, 1000.f );
    }
    else if( name == "orthographic" )
    {
        PBRT_WARNING( "Orthographic camera not implemented." );
    }
    else if( name == "environment" )
    {
        PBRT_WARNING( "Environment camera not implemented." );
    }
    else if( name == "realistic" )
    {
        PBRT_WARNING( "Realistic camera not implemented." );
    }
    else
    {
        PBRT_WARNING( "Unknown camera type '" + name + "'" );
    }
}

void PbrtApiImpl::makeNamedMedium( const std::string& name, const ParamSet& params )
{
    PBRT_WARNING( "MakeNamedMedium '" + name + "' not implemented." );
}

void PbrtApiImpl::mediumInterface( const std::string& insideName, const std::string& outsideName )
{
    PBRT_WARNING( "MediumInterface '" + insideName + "', '" + outsideName + "' not implemented." );
}

void PbrtApiImpl::worldBegin()
{
    m_inWorld          = true;
    m_currentTransform = ::pbrt::Transform();
}

void PbrtApiImpl::attributeBegin()
{
    requireInWorld( "AttributeBegin" );
    m_transformStack.push_back( m_currentTransform );
    m_graphicsStateStack.push_back( m_graphicsState );
}

void PbrtApiImpl::attributeEnd()
{
    requireInWorld( "AttributeEnd" );
    if( m_transformStack.empty() )
    {
        PBRT_ERROR( "AttributeEnd has no corresponding AttributeBegin" );
        return;
    }
    m_currentTransform = m_transformStack.back();
    m_transformStack.pop_back();
    m_graphicsState = m_graphicsStateStack.back();
    m_graphicsStateStack.pop_back();
}

void PbrtApiImpl::transformBegin()
{
    requireInWorld( "TransformBegin" );
    m_transformStack.push_back( m_currentTransform );
}

void PbrtApiImpl::transformEnd()
{
    requireInWorld( "TransformEnd" );
    if( m_transformStack.empty() )
    {
        PBRT_ERROR( "Too many TransformEnd for TransformBegin (" + std::to_string( m_transformStack.size() )
               + " in excess)" );
        return;
    }
    m_currentTransform = m_transformStack.back();
    m_transformStack.pop_back();
}

void PbrtApiImpl::texture( const std::string& name, const std::string& type, const std::string& tex_type, const ParamSet& params )
{
    requireInWorld( "Texture" );
    if( type == "float" )
    {
        m_graphicsState.floatTextures[name] = TextureDefinition{ tex_type, params };
    }
    else if( type == "color" || type == "spectrum" )
    {
        m_graphicsState.spectrumTextures[name] = TextureDefinition{ tex_type, params };
    }
    else
    {
        PBRT_ERROR( "Texture " + name + ' ' + type + ' ' + tex_type + " not implemented" );
    }
}

void PbrtApiImpl::material( const std::string& name, const ParamSet& params )
{
    requireInWorld( "Material" );
    m_graphicsState.currentMaterial.name   = name;
    m_graphicsState.currentMaterial.params = params;
    m_graphicsState.currentNamedMaterial.clear();
}

void PbrtApiImpl::makeNamedMaterial( const std::string& name, const ParamSet& params )
{
    requireInWorld( "MakeNamedMaterial" );
    m_graphicsState.namedMaterials[name] = MaterialDefinition{ name, params };
}

void PbrtApiImpl::namedMaterial( const std::string& name )
{
    requireInWorld( "NamedMaterial" );
    auto it = m_graphicsState.namedMaterials.find( name );
    if( it == m_graphicsState.namedMaterials.end() )
    {
        PBRT_WARNING( "Unknown named material '" + name + "'" );
        return;
    }
    m_graphicsState.currentNamedMaterial = name;
    m_graphicsState.currentMaterial      = it->second;
}

void PbrtApiImpl::lightSource( const std::string& name, const ParamSet& params )
{
    requireInWorld( "LightSource" );
    const auto lookupSpectrum = [&]( const char* spectrumName ) {
        ::pbrt::Spectrum spectrum{ params.FindOneSpectrum( spectrumName, ::pbrt::Spectrum( 1.0f ) ) };
        float            rgb[3];
        spectrum.ToRGB( rgb );
        return ::pbrt::Point3f( rgb[0], rgb[1], rgb[2] );
    };
    if( name == "distant" )
    {
        const ::pbrt::Point3f colorScale{ lookupSpectrum( "scale" ) };
        const ::pbrt::Point3f color{ lookupSpectrum( "L" ) };
        const auto            lookupPoint = [&]( const char* pointName, const ::pbrt::Point3f& fallback ) {
            return params.FindOnePoint3f( pointName, fallback );
        };
        const ::pbrt::Vector3f direction{ lookupPoint( "from", ::pbrt::Point3f( 0.0f, 0.0f, 0.0f ) )
                                          - lookupPoint( "to", ::pbrt::Point3f( 0.0f, 0.0f, 1.0f ) ) };
        m_scene->distantLights.push_back( DistantLightDefinition{ colorScale, color, direction, m_currentTransform } );
    }
    else if( name == "infinite" )
    {
        const ::pbrt::Point3f colorScale{ lookupSpectrum( "scale" ) };
        const ::pbrt::Point3f color{ lookupSpectrum( "L" ) };
        const int             shadowSamples = params.FindOneInt( "samples", params.FindOneInt( "nsamples", 1 ) );
        const std::string mapName = params.FindOneFilename( "mapname", "" );
        if(shadowSamples != 1)
        {
            PBRT_WARNING( R"(LightSource "infinite" "integer samples" not implemented)" );
        }
        auto existingInfiniteLightHasEnvironmentMap = [&] {
            return std::find_if( m_scene->infiniteLights.begin(), m_scene->infiniteLights.end(),
                                 []( const InfiniteLightDefinition& light ) { return !light.environmentMapName.empty(); } )
                   != m_scene->infiniteLights.end();
        };
        if( !mapName.empty() && existingInfiniteLightHasEnvironmentMap() )
        {
            PBRT_WARNING( R"(LightSource "infinite" "string mapname" seen more than once; only one environment map is supported)" );
        }
        m_scene->infiniteLights.push_back( InfiniteLightDefinition{ colorScale, color, shadowSamples, mapName, m_currentTransform } );
    }
    else
    {
        PBRT_WARNING( "LightSource '" + name + "' not implemented." );
    }
}

void PbrtApiImpl::areaLightSource( const std::string& name, const ParamSet& params )
{
    requireInWorld( "AreaLightSource" );
    PBRT_WARNING( "AreaLightSource not implemented." );
}

void PbrtApiImpl::addShape( ShapeDefinition shape )
{
    if( shape.bounds == ::pbrt::Bounds3f() )
    {
        return;
    }

    m_currentBounds = Union( m_currentBounds, m_currentTransform( shape.bounds ) );
    if( m_currentObjectName.empty() )
    {
        m_scene->freeShapes.emplace_back( std::move( shape ) );
    }
    else
    {
        m_scene->objectShapes[m_currentObjectName].emplace_back( std::move( shape ) );
    }
}

void PbrtApiImpl::shape( const std::string& type, const ParamSet& params )
{
    requireInWorld( "Shape" );
    if( type == "plymesh" )
    {
        addShape( createPlyMesh( params ) );
    }
    else if( type == "trianglemesh" )
    {
        addShape( createTriangleMesh( params ) );
    }
    else if( type == "sphere" )
    {
        addShape( createSphere( params ) );
    }
    else if( type == "cone" || type == "curve" || type == "cylinder" || type == "disk" || type == "hyperboloid"
             || type == "paraboloid" || type == "heightfield" || type == "loopsubdiv" || type == "nurbs" )
    {
#ifndef NDEBUG
        PBRT_WARNING( "Unsupported shape type " + type );
#endif        
    }
    else
    {
        PBRT_ERROR( "Unknown shape type " + type );
    }
}

void PbrtApiImpl::reverseOrientation()
{
    requireInWorld( "ReverseOrientation" );
    m_graphicsState.reverseOrientation = true;
    PBRT_WARNING( "ReverseOrientation not implemented." );
}

void PbrtApiImpl::objectBegin( const std::string& name )
{
    requireInWorld( "ObjectBegin" );
    m_currentObjectName = name;
    m_boundsStack.push_back( m_currentBounds );
    m_currentBounds = ::pbrt::Bounds3f{};
    attributeBegin();
}

void PbrtApiImpl::objectEnd()
{
    requireInWorld( "ObjectEnd" );
    if( m_boundsStack.empty() )
    {
        PBRT_ERROR( "Too many ObjectEnd for ObjectBegin (" + std::to_string( m_boundsStack.size() ) + " in excess)" );
    }
    // if object is empty, current bounds are invalid bounds
    m_scene->objects[m_currentObjectName] = ObjectDefinition{ m_currentTransform, m_currentBounds };
    m_currentObjectName.clear();
    m_currentBounds = m_boundsStack.back();
    m_boundsStack.pop_back();
    attributeEnd();
}

void PbrtApiImpl::objectInstance( const std::string& name )
{
    requireInWorld( "ObjectInstance" );
    const auto it = m_scene->objects.find( name );
    if( it == m_scene->objects.end() )
    {
        PBRT_ERROR( "Unknown object " + name );
        return;
    }

    // No point in instantiating empty objects.
    if( m_scene->objectShapes[name].empty() )
    {
        static std::vector<std::string> warnedInstances;

        // Only warn once.
        auto pos = std::find( warnedInstances.begin(), warnedInstances.end(), name );
        if( pos == warnedInstances.end() )
        {
            warnedInstances.push_back( name );
            PBRT_WARNING( "Skipping instances of empty object " + name );
        }
        return;
    }

    ++m_scene->instanceCounts[name];
    ::pbrt::Bounds3f objectBounds = it->second.transform( it->second.bounds );
    m_scene->objectInstances.emplace_back( ObjectInstanceDefinition{ name, m_currentTransform, objectBounds } );
    const ::pbrt::Bounds3f instanceBounds{ m_currentTransform( objectBounds ) };
    m_currentBounds = Union( m_currentBounds, instanceBounds );
}

void PbrtApiImpl::worldEnd()
{
    requireInWorld( "WorldEnd" );
    m_inWorld       = false;
    m_scene->bounds = m_currentBounds;
}

void PbrtApiImpl::requireInWorld( const char* name )
{
    if( !m_inWorld )
    {
        PBRT_ERROR( std::string{ name } + " can only be used inside a World block." );
    }
}

ShapeDefinition PbrtApiImpl::createPlyMesh( const ::pbrt::ParamSet& params )
{
    const std::string filename = ::pbrt::ResolveFilename( params.FindOneFilename( "filename", "" ) );
    PBRT_INFO( std::string( "Reading info from " + filename ) );
    const MeshInfo         info     = m_infoReader->read( filename );
    const ::pbrt::Bounds3f bounds{ ::pbrt::Point3f( info.minCoord[0], info.minCoord[1], info.minCoord[2] ),
                                   ::pbrt::Point3f( info.maxCoord[0], info.maxCoord[1], info.maxCoord[2] ) };

    return { "plymesh",
             m_currentTransform,
             getShapeMaterial( params ),
             bounds,
             PlyMeshData{ filename, m_infoReader->getLoader( filename ) },
             TriangleMeshData{},
             SphereData{} };
}

ShapeDefinition PbrtApiImpl::createTriangleMesh( const ::pbrt::ParamSet& params )
{
    TriangleMeshData       data{};
    int                    numIndices{};
    const int*             indices = params.FindInt( "indices", &numIndices );
    int                    numVertices{};
    const ::pbrt::Point3f* vertices = params.FindPoint3f( "P", &numVertices );
    std::copy( indices, indices + numIndices, std::back_inserter( data.indices ) );
    ::pbrt::Bounds3f bounds{};
    std::transform( vertices, vertices + numVertices, std::back_inserter( data.points ), [&bounds]( const ::pbrt::Point3f& pt ) {
        bounds = Union( bounds, pt );
        return pt;
    } );
    int                    numNormals{};
    const ::pbrt::Point3f* normals = params.FindPoint3f( "N", &numNormals );
    std::copy( normals, normals + numNormals, std::back_inserter( data.normals ) );
    int          numUvs{};
    const float* uvs = params.FindFloat( "uv", &numUvs );
    for( int i = 0; i < numUvs / 2; ++i )
    {
        data.uvs.push_back( ::pbrt::Point2f( uvs[i * 2], uvs[i * 2 + 1] ) );
    }
    return { "trianglemesh",    m_currentTransform, getShapeMaterial( params ), bounds, PlyMeshData{},
             std::move( data ), SphereData{} };
}

ShapeDefinition PbrtApiImpl::createSphere( const ::pbrt::ParamSet& params )
{
    SphereData data{};

    auto getSingleFloat = [&]( const char* name, float defaultValue ) {
        int          numFloats{};
        const float* value = params.FindFloat( name, &numFloats );
        return numFloats > 0 && value != nullptr ? *value : defaultValue;
    };
    data.radius = getSingleFloat( "radius", 1.0f );
    data.zMin   = getSingleFloat( "zmin", -data.radius );
    data.zMax   = getSingleFloat( "zmax", data.radius );
    data.phiMax = getSingleFloat( "phimax", 360.0f );
    return { "sphere",
             m_currentTransform,
             getShapeMaterial( params ),
             ::pbrt::Bounds3f{ ::pbrt::Point3f{ data.radius, data.radius, data.radius },
                               ::pbrt::Point3f{ -data.radius, -data.radius, -data.radius } },
             PlyMeshData{},
             TriangleMeshData{},
             data };
}

PlasticMaterial PbrtApiImpl::getShapeMaterial( const ::pbrt::ParamSet& params ) const
{
    // Default to no ambient, all diffuse, no specular, if not specified
    const ::pbrt::Point3f zero{};
    const ::pbrt::Point3f Ka                  = lookupParam( "Ka", params, zero );
    const ::pbrt::Point3f Kd                  = lookupParam( "Kd", params, ::pbrt::Point3f( 1.0f, 1.0f, 1.0f ) );
    const ::pbrt::Point3f Ks                  = lookupParam( "Ks", params, zero );
    const std::string     alphaMapFileName    = lookupFloatTextureFileName( "alpha", params );
    const std::string     diffuseMapFileName  = lookupSpectrumTextureFileName( "Kd", params );
    const std::string     specularMapFileName = lookupSpectrumTextureFileName( "Ks", params );
    return { Ka, Kd, Ks, alphaMapFileName, diffuseMapFileName, specularMapFileName };
}

static bool findParamInSet( const std::string& name, const ::pbrt::ParamSet& params, ::pbrt::Point3f& result )
{
    int num_values{};
    // Look for a spectrum param first
    const ::pbrt::Spectrum* spec = params.FindSpectrum( name, &num_values );
    if( num_values > 0 )
    {
        float rgb[3] = { 0.0f };
        spec->ToRGB( &rgb[0] );
        result = ::pbrt::Point3f( rgb[0], rgb[1], rgb[2] );
    }
    else
    {
        // Look for a point param
        const ::pbrt::Point3f* pt = params.FindPoint3f( name, &num_values );
        if( num_values > 0 )
        {
            result = *pt;
        }
    }
    return num_values > 0;
}

bool PbrtApiImpl::findParam( const std::string& name, const PbrtApiImpl::GraphicsState& state, const MaterialDefinition& material, ::pbrt::Point3f& result ) const
{
    if( material.name == "mix" )
    {
        const std::string name1{ material.params.FindOneString( "namedmaterial1", std::string{} ) };
        const std::string name2{ material.params.FindOneString( "namedmaterial2", std::string{} ) };
        const auto        it1 = state.namedMaterials.find( name1 );
        const auto        it2 = state.namedMaterials.find( name2 );
        if( it1 != state.namedMaterials.end() && it2 != state.namedMaterials.end() )
        {
            const MaterialDefinition mat1{ it1->second };
            const MaterialDefinition mat2{ it2->second };
            const std::string type1{mat1.params.FindOneString("type", std::string{})};
            const std::string type2{mat2.params.FindOneString("type", std::string{})};
            if( type1 != "translucent" && type2 == "translucent" )
            {
                return findParam( name, state, mat1, result );
            }
        }
    }
    return findParamInSet( name, material.params, result );
}

bool PbrtApiImpl::findParam( const std::string& name, const GraphicsState& state, ::pbrt::Point3f& result ) const
{
    if( !state.currentNamedMaterial.empty() )
    {
        auto it = state.namedMaterials.find( state.currentNamedMaterial );
        if( it != state.namedMaterials.end() )
        {
            return findParam(name, state, it->second, result);
        }
    }
    if( !state.currentMaterial.name.empty() )
    {
        return findParam( name, state, state.currentMaterial, result);
    }
    return false;
}

::pbrt::Point3f PbrtApiImpl::lookupParam( const std::string& name, const ::pbrt::ParamSet& params, ::pbrt::Point3f def ) const
{
    ::pbrt::Point3f result;
    if( findParamInSet( name, params, result ) )
    {
        return result;
    }
    if( findParam( name, m_graphicsState, result ) )
    {
        return result;
    }
    for( std::size_t i = m_graphicsStateStack.size(); i > 0; --i )
    {
        if( findParam( name, m_graphicsStateStack[i - 1], result ) )
        {
            return result;
        }
    }
    return def;
}

std::string PbrtApiImpl::findTexture( const std::string& name, const GraphicsState& state, const MaterialDefinition& material ) const
{
    if( material.name == "mix" )
    {
        const std::string name1{ material.params.FindOneString( "namedmaterial1", std::string{} ) };
        const std::string name2{ material.params.FindOneString( "namedmaterial2", std::string{} ) };
        const auto        it1 = state.namedMaterials.find( name1 );
        const auto        it2 = state.namedMaterials.find( name2 );
        if( it1 != state.namedMaterials.end() && it2 != state.namedMaterials.end() )
        {
            const MaterialDefinition mat1{ it1->second };
            const MaterialDefinition mat2{ it2->second };
            const std::string        type1{ mat1.params.FindOneString( "type", std::string{} ) };
            const std::string        type2{ mat2.params.FindOneString( "type", std::string{} ) };
            if( type1 != "translucent" && type2 == "translucent" )
            {
                return findTexture( name, state, mat1 );
            }
        }
    }
    return material.params.FindTexture( name );
}

std::string PbrtApiImpl::findTexture( const std::string& name, const GraphicsState& state ) const
{
    std::string result;
    if( !state.currentNamedMaterial.empty() )
    {
        auto it = state.namedMaterials.find( state.currentNamedMaterial );
        if( it != state.namedMaterials.end() )
        {
            return findTexture( name, state, it->second );
        }
    }
    if (!state.currentMaterial.name.empty())
    {
        return findTexture( name, state, state.currentMaterial );
    }
    return {};
}

void PbrtApiImpl::resetState()
{
    m_currentTransform = ::pbrt::Transform{};
    m_coordinateSystems.clear();
    m_transformStack.clear();
    m_graphicsState = GraphicsState{};
    m_graphicsStateStack.clear();
    m_objectName.clear();
    m_currentObjectName.clear();
    m_currentBounds = ::pbrt::Bounds3f{};
    m_boundsStack.clear();
    m_inWorld       = false;
    m_scene         = std::make_shared<SceneDescription>();
    m_scene->bounds = ::pbrt::Bounds3f{};
}

std::string PbrtApiImpl::lookupTextureName( const std::string& name, const ::pbrt::ParamSet& params ) const
{
    std::string textureName = params.FindTexture( name );
    if( !textureName.empty() )
    {
        return textureName;
    }

    textureName = findTexture( name, m_graphicsState );
    if( !textureName.empty() )
    {
        return textureName;
    }

    for( std::size_t i = m_graphicsStateStack.size(); i > 0; --i )
    {
        textureName = findTexture( name, m_graphicsStateStack[i - 1] );
        if( !textureName.empty() )
        {
            return textureName;
        }
    }

    return textureName;
}

std::string PbrtApiImpl::lookupSpectrumTextureFileName( const std::string& name, const ::pbrt::ParamSet& params ) const
{
    const std::string textureName = lookupTextureName( name, params );
    if( textureName.empty() )
    {
        return {};
    }

    auto it = m_graphicsState.spectrumTextures.find( textureName );
    if( it != m_graphicsState.spectrumTextures.end() )
    {
        return it->second.params.FindOneFilename( "filename", "" );
    }
    for( std::size_t i = m_graphicsStateStack.size() - 1; i > 0; --i )
    {
        it = m_graphicsStateStack[i - 1].spectrumTextures.find( textureName );
        if( it != m_graphicsStateStack[i - 1].spectrumTextures.end() )
        {
            return it->second.params.FindOneFilename( "filename", "" );
        }
    }
    return {};
}

std::string PbrtApiImpl::lookupFloatTextureFileName( const std::string& name, const ::pbrt::ParamSet& params ) const
{
    const std::string texture_name = lookupTextureName( name, params );
    if( texture_name.empty() )
    {
        return {};
    }

    auto it = m_graphicsState.floatTextures.find( texture_name );
    if( it != m_graphicsState.floatTextures.end() )
    {
        return it->second.params.FindOneFilename( "filename", "" );
    }
    for( std::size_t i = m_graphicsStateStack.size() - 1; i > 0; --i )
    {
        it = m_graphicsStateStack[i - 1].floatTextures.find( texture_name );
        if( it != m_graphicsStateStack[i - 1].floatTextures.end() )
        {
            return it->second.params.FindOneFilename( "filename", "" );
        }
    }
    return {};
}

}  // namespace pbrt
}  // namespace otk
