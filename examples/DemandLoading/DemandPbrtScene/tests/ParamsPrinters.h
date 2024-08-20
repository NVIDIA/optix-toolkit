// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <Params.h>

#include <OptiXToolkit/ShaderUtil/vec_printers.h>

#include <vector_types.h>

#include <iostream>

namespace demandPbrtScene {

inline std::ostream& operator<<( std::ostream& str, const DirectionalLight& value )
{
    return str << "DirectionalLight{ dir: " << value.direction << ", color: " << value.color << " }";
}

inline std::ostream& operator<<( std::ostream& str, const InfiniteLight& value )
{
    return str << "InfiniteLight{ color: " << value.color << ", scale: " << value.scale
               << ", textureId: " << value.skyboxTextureId << " }";
}

inline std::ostream& operator<<( std::ostream& str, const TriangleNormals& value )
{
    return str << "0: " << value.N[0] << ", 1: " << value.N[1] << ", 2: " << value.N[2];
}

inline std::ostream& operator<<( std::ostream& str, const TriangleUVs& value )
{
    return str << "0: " << value.UV[0] << ", 1: " << value.UV[1] << ", 2: " << value.UV[2];
}

inline std::ostream& operator<<( std::ostream& str, const PartialMaterial& value )
{
    return str << "PartialMaterial{ " << value.alphaTextureId << " }";
}

inline std::ostream& operator<<( std::ostream& str, const MaterialFlags val )
{
    str << "MaterialFlags{ ";
    bool emitted{ false };
    auto emitSep = [&]() -> std::ostream& {
        if( emitted )
        {
            str << " | ";
        }
        emitted = true;
        return str;
    };
    if( flagSet( val, MaterialFlags::ALPHA_MAP ) )
    {
        emitSep() << "AlphaMap";
    }
    if( flagSet( val, MaterialFlags::ALPHA_MAP_ALLOCATED ) )
    {
        emitSep() << "AlphaMapAllocated";
    }
    if( flagSet( val, MaterialFlags::DIFFUSE_MAP ) )
    {
        emitSep() << "DiffuseMap";
    }
    if( flagSet( val, MaterialFlags::DIFFUSE_MAP_ALLOCATED ) )
    {
        emitSep() << "DiffuseMapAllocated";
    }
    if( !emitted )
    {
        str << "None";
    }
    return str << " }";
}

inline std::ostream& operator<<( std::ostream& str, const PhongMaterial& val )
{
    return str << "PhongMaterial{ "                          //
               << "Ka" << val.Ka                             //
               << ", Kd" << val.Kd                           //
               << ", Ks" << val.Ks                           //
               << ", Kr" << val.Kr                           //
               << ", exp " << val.phongExp                   //
               << ", flags " << val.flags                    //
               << ", alphaTexId " << val.alphaTextureId      //
               << ", diffuseTexId " << val.diffuseTextureId  //
               << " }";
}

inline bool operator==( const PrimitiveMaterialRange& lhs, const PrimitiveMaterialRange& rhs )
{
    return lhs.primitiveEnd == rhs.primitiveEnd  //
           && lhs.materialId == rhs.materialId;
}

inline bool operator!=( const PrimitiveMaterialRange& lhs, const PrimitiveMaterialRange& rhs )
{
    return !( lhs == rhs );
}

inline std::ostream& operator<<( std::ostream& str, const PrimitiveMaterialRange& value )
{
    return str << "PrimitiveMaterialRange{ " << value.primitiveEnd << ", " << value.materialId << " }";
}

inline bool operator==( const MaterialIndex& lhs, const MaterialIndex& rhs )
{
    return lhs.numPrimitiveGroups == rhs.numPrimitiveGroups  //
           && lhs.primitiveMaterialBegin == rhs.primitiveMaterialBegin;
}

inline bool operator!=( const MaterialIndex& lhs, const MaterialIndex& rhs )
{
    return !( lhs == rhs );
}

inline std::ostream& operator<<( std::ostream& str, const MaterialIndex& value )
{
    return str << "MaterialIndex{ " << value.numPrimitiveGroups << ", " << value.primitiveMaterialBegin << " }";
}

}  // namespace demandPbrtScene
