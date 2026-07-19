// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/PbrtCheckerboardImageSource.h"

#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <cuda_runtime.h>

#include <initializer_list>
#include <string>

namespace demandPbrtScene {

inline MaterialFlags plasticMaterialFlags( const otk::pbrt::PlasticMaterial& material )
{
    MaterialFlags flags{};
    if( !material.alphaMapFileName.empty() )
        flags |= MaterialFlags::ALPHA_MAP;
    if( !material.diffuseMapFileName.empty() )
        flags |= MaterialFlags::DIFFUSE_MAP;
    return flags;
}

inline std::string pbrtTextureGraphKey( const std::string& valueType, const std::string& textureName )
{
    return valueType + ":" + textureName;
}

inline const otk::pbrt::PbrtTexture* findPbrtTexture( const otk::pbrt::PbrtMaterial&     material,
                                                      const std::string&                 textureName,
                                                      std::initializer_list<const char*> valueTypes )
{
    if( textureName.empty() )
    {
        return nullptr;
    }

    for( const char* valueType : valueTypes )
    {
        const auto it = material.graph.textures.find( pbrtTextureGraphKey( valueType, textureName ) );
        if( it != material.graph.textures.end() )
        {
            return &it->second;
        }
    }
    return nullptr;
}

inline std::string pbrtTextureMapName( const otk::pbrt::PbrtTexture* texture )
{
    if( texture == nullptr )
    {
        return {};
    }
    if( texture->type == "imagemap" )
    {
        return texture->params.FindOneFilename( "filename", "" );
    }
    if( texture->type == "checkerboard" )
    {
        return pbrtCheckerboardTextureKey( *texture );
    }
    return {};
}

inline std::string pbrtColorMapFileName( const otk::pbrt::PbrtMaterial& material, const char* paramName )
{
    const std::string textureName{ material.params.FindTexture( paramName ) };
    return pbrtTextureMapName( findPbrtTexture( material, textureName, { "spectrum", "color" } ) );
}

inline std::string pbrtDiffuseMapFileName( const otk::pbrt::PbrtMaterial& material )
{
    return pbrtColorMapFileName( material, "Kd" );
}

inline std::string pbrtMirrorReflectanceMapFileName( const otk::pbrt::PbrtMaterial& material )
{
    return material.type == "mirror" ? pbrtColorMapFileName( material, "Kr" ) : std::string{};
}

inline std::string pbrtAlphaMapFileName( const otk::pbrt::PbrtMaterial& material )
{
    for( const char* paramName : { "alpha", "shadowalpha", "opacity" } )
    {
        const std::string textureName{ material.params.FindTexture( paramName ) };
        const std::string fileName{ pbrtTextureMapName( findPbrtTexture( material, textureName, { "float" } ) ) };
        if( !fileName.empty() )
        {
            return fileName;
        }
    }
    return {};
}

inline otk::pbrt::PlasticMaterial fallbackMaterialForShape( const otk::pbrt::ShapeDefinition& shape )
{
    otk::pbrt::PlasticMaterial result{ shape.material };

    const std::string diffuseMapFileName{ pbrtDiffuseMapFileName( shape.pbrtMaterial ) };
    if( !diffuseMapFileName.empty() )
    {
        result.diffuseMapFileName = diffuseMapFileName;
    }
    else
    {
        const std::string reflectanceMapFileName{ pbrtMirrorReflectanceMapFileName( shape.pbrtMaterial ) };
        if( !reflectanceMapFileName.empty() )
        {
            result.diffuseMapFileName = reflectanceMapFileName;
        }
    }

    const std::string alphaMapFileName{ pbrtAlphaMapFileName( shape.pbrtMaterial ) };
    if( !alphaMapFileName.empty() )
    {
        result.alphaMapFileName = alphaMapFileName;
    }

    return result;
}

inline MaterialFlags shapeMaterialFlags( const otk::pbrt::ShapeDefinition& shape )
{
    return plasticMaterialFlags( fallbackMaterialForShape( shape ) );
}

}  // namespace demandPbrtScene
