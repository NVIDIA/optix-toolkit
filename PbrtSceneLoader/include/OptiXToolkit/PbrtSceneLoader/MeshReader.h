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

#include <memory>
#include <string>
#include <vector>

namespace otk {
namespace pbrt {

struct MeshInfo
{
    long  numVertices;
    long  numNormals;
    long  numTextureCoordinates;
    long  numTriangles;
    float minCoord[3];
    float maxCoord[3];
};

struct MeshData
{
    std::vector<float> vertexCoords;  // 3 floats per vertex
    std::vector<int>   indices;       // 3 indices per face
    std::vector<float> normalCoords;  // 3 floats per vertex
    std::vector<float> uvCoords;      // 2 floats per vertex
};

class MeshLoader
{
public:
    virtual ~MeshLoader() = default;

    virtual MeshInfo getMeshInfo() const = 0;

    virtual void load( MeshData& buffers ) = 0;
};

using MeshLoaderPtr = std::shared_ptr<MeshLoader>;

class MeshInfoReader
{
  public:
    virtual ~MeshInfoReader() = default;

    virtual MeshInfo read( const std::string& filename ) = 0;

    virtual MeshLoaderPtr getLoader( const std::string& filename ) = 0;
};

}  // namespace pbrt
}  // namespace otk
