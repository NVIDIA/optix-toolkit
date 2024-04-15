//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
    bool emitted{false};
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
    return str << "PhongMaterial{ Ka" << val.Ka << ", Kd" << val.Kd << ", Ks" << val.Ks << ", Kr" << val.Kr << ", exp "
        << val.phongExp << ", flags " << val.flags << ", alphaTexId " << val.alphaTextureId << ", diffuseTexId "
        << val.diffuseTextureId << " }";
}

}  // namespace demandPbrtScene
