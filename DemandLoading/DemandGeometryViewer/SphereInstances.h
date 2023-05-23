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

#include <OptiXToolkit/Memory/DeviceBuffer.h>
#include <OptiXToolkit/Memory/SyncVector.h>

#include <optix.h>

#include <cuda.h>

#include <cstdint>

namespace demandGeometryViewer {

using uint_t = unsigned int;

class SphereInstances
{
  public:
    void add( float3 center, float radius, int index );
    void remove( int index );
    void setSbtIndex( uint_t index ) { m_sbtIndex = index; }

    OptixTraversableHandle createTraversable( OptixDeviceContext dc, CUstream stream );

    const uint_t* getIndicesDevicePtr() const { return m_indices.typedDevicePtr(); }

  private:
    otk::SyncVector<uint_t>        m_indices;
    otk::SyncVector<float3>        m_centers;
    otk::SyncVector<float>         m_radii;
    otk::SyncVector<std::uint32_t> m_sbtIndices;
    otk::DeviceBuffer              m_devTempBufferGas;
    otk::DeviceBuffer              m_devGeomAs;
    uint_t                         m_sbtIndex{};
};

}  // namespace demandGeometryViewer
