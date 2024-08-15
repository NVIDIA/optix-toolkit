// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
