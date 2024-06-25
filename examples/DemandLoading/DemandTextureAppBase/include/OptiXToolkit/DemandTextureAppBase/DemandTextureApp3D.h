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

