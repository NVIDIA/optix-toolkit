// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/PbrtApi/PbrtApi.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <core/api.h>
#include <core/geometry.h>
#include <core/paramset.h>
#include <core/transform.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

// Follow the semantics as described at https://pbrt.org/fileformat-v3

namespace otk {
namespace pbrt {

class Logger;
class MeshInfoReader;

class PbrtApiImpl : public Api
{
public:
    PbrtApiImpl( const char* programName, std::shared_ptr<Logger> logger, std::shared_ptr<MeshInfoReader> infoReader );
    ~PbrtApiImpl() override;

    SceneDescriptionPtr parseFile( const std::string& filename );
    SceneDescriptionPtr parseString( const std::string& str );

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

}  // namespace pbrt
}  // namespace otk
