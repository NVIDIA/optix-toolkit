// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
