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

