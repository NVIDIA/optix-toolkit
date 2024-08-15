// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandTextureAppBase/DemandTextureApp.h>
#include <OptiXToolkit/DemandTextureAppBase/ShapeMaker.h>

namespace demandTextureApp
{

class DemandTextureApp3D : public DemandTextureApp
{
  public:
    DemandTextureApp3D( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop );
    virtual ~DemandTextureApp3D() {}

    void buildAccel( PerDeviceOptixState& state ) override;
    void createSBT( PerDeviceOptixState& state ) override;

  protected:
    std::vector<float4> m_vertices;
    std::vector<float3> m_normals;
    std::vector<float2> m_tex_coords;
    std::vector<uint32_t> m_material_indices;
    std::vector<TriangleHitGroupData> m_materials;
    
    SurfaceTexture makeSurfaceTex( int kd, int kdtex, int ks, int kstex, int kt, int kttex, float roughness, float ior );
    void addShapeToScene( std::vector<Vert>& shape, unsigned int materialId );
    void copyGeometryToDevice();

    void cursorPosCallback( GLFWwindow* window, double xpos, double ypos ) override;
    void pollKeys() override;
};

} // namespace demandTextureApp
