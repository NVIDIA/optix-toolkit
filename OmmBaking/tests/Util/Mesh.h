// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "Util/Exception.h"
#include "Util/VecMath.h"

#include <OptiXToolkit/CuOmmBaking/CuBuffer.h>
#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>

struct Mesh_format
{
    uint32_t                  texCoordStrideInBytes       = 0;
    uint32_t                  verticesStrideInBytes       = 0;
    uint32_t                  indicesStrideInBytes        = 0;
    cuOmmBaking::IndexFormat  indexFormat                 = cuOmmBaking::IndexFormat::I32_UINT;
    uint32_t                  textureIndicesStrideInBytes = 0;
    cuOmmBaking::IndexFormat  textureIndexFormat          = cuOmmBaking::IndexFormat::I32_UINT;
};


class Mesh
{
  public:
    using Format = Mesh_format;

    Mesh() = default;
    ~Mesh() { destroy(); }

    Mesh( const Mesh& source ) = delete;
    Mesh( Mesh&& source ) noexcept { swap( source ); }

    Mesh& operator=( const Mesh& source ) = delete;

    Mesh& operator=( Mesh&& source ) noexcept
    {
        swap( source );
        return *this;
    }

    void destroy()
    {
        m_indices.free();
        m_vertices.free();
        m_texCoords.free();
        m_textures.free();

        m_format = {};

        m_numVertices  = 0;
        m_numTexCoords = 0;
        m_numIndices   = 0;
    }

    cudaError_t create( float3 vertMin = { 0, 0, 0 }, float3 vertMax = { 1, 1, 0 }, float2 uvMin = { 0, 0 }, float2 uvMax = { 1, 1 }, uint2 gridRes = { 1, 1 }, unsigned numTextures = 1, Format format = {} );
    cudaError_t create( const std::vector<float2>& texCoords, const std::vector<float3>& vertices, const std::vector<uint3>& indices, std::vector<unsigned> textures, Format format = {} );

    cuOmmBaking::BakeInputDesc getBakingInputDesc() const;

  private:
    void swap( Mesh& source ) noexcept
    {
        std::swap( source.m_indices, m_indices );
        std::swap( source.m_vertices, m_vertices );
        std::swap( source.m_texCoords, m_texCoords );
        std::swap( source.m_textures, m_textures );

        std::swap( source.m_format, m_format );

        std::swap( source.m_numVertices, m_numVertices );
        std::swap( source.m_numTexCoords, m_numTexCoords );
        std::swap( source.m_numIndices, m_numIndices );
    }

    CuBuffer<char> m_indices;
    CuBuffer<char> m_vertices;
    CuBuffer<char> m_texCoords;
    CuBuffer<char> m_textures;

    Format   m_format       = {};
    uint32_t m_numVertices  = 0;
    uint32_t m_numTexCoords = 0;
    uint32_t m_numIndices   = 0;
};
