// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda_runtime.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <vector>
using namespace otk;

struct Vert
{
    float3 p; // position
    float3 n; // normal
    float2 t; // tex coord
};

class ShapeMaker
{
  public:
    static void makeAxisPlane( float3 minCorner, float3 maxCorner, std::vector<Vert>& shape );
    static void makeCircle( float3 center, float radius, int numSegments, std::vector<Vert>& shape );
    static void makeSphere( float3 center, float radius, int numSegments, std::vector<Vert>& shape, float beginAngle=0.0f, float endAngle=M_PIf );
    static void makeCylinder( float3 basePoint, float radius, float height, int numSegments, std::vector<Vert>& shape );
    static void makeCone( float3 basePoint, float radius, float height, int numSegments, std::vector<Vert>& shape );
    static void makeTorus( float3 center, float radius1, float radius2, int numSegments, std::vector<Vert>& shape );
    static void makeVase( float3 basePoint, float radius1, float radius2, float height, int numSegments, std::vector<Vert>& shape );
    static void makeBox( float3 corner, float3 dim, std::vector<Vert>& shape );

    static void spinZaxis( std::vector<Vert>& silhouette, int numSegments, float3 translation, std::vector<Vert>& shape );
    static Vert rotateSilhouettePoint( const Vert& p, float angle );
};

