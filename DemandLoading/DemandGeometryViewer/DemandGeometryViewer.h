//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/DemandGeometry/DemandGeometry.h>
#include <OptiXToolkit/ShaderUtil/DebugLocation.h>

#include <optix_types.h>

#include <vector_types.h>

namespace demandGeometryViewer {

using uint_t = unsigned int;

enum RayType
{
    RAYTYPE_RADIANCE = 0,
    RAYTYPE_COUNT
};

struct CameraData
{
    float3 eye;
    float3 U;
    float3 V;
    float3 W;
};

struct MissData
{
    float3 background;
};

struct PhongMaterial
{
    float3 Ka;
    float3 Kd;
    float3 Ks;
    float3 Kr;
    float  phongExp;
};

struct HitGroupData
{
    PhongMaterial material;
};

struct BasicLight
{
    float3 pos;
    float3 color;
};

struct GetSphereData
{
    bool          useOptixGetSphereData;
    const float3* centers;
    const float*  radii;
};

struct Params
{
    uchar4*                      image;
    uint_t                       width;
    uint_t                       height;
    BasicLight                   lights[3];
    float3                       ambientColor;
    float3                       proxyFaceColors[6];
    float                        sceneEpsilon;
    OptixTraversableHandle       traversable;
    demandLoading::DeviceContext demandContext;
    demandGeometry::Context      demandGeomContext;
    const uint_t*                demandMaterialPageIds;
    float3                       demandMaterialColor;
    const uint_t*                sphereIds;
    GetSphereData                getSphereData;
    otk::DebugLocation           debug;
};

}  // namespace demandGeometryViewer
