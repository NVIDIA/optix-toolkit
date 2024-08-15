// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
    void add( float3 center, float radius, uint_t id, uint_t sbtIndex );
    void remove( uint_t id );
    void setInstanceSbtIndex( uint_t id, uint_t index );

    OptixTraversableHandle createTraversable( OptixDeviceContext dc, CUstream stream );

    const uint_t* getSphereIdsDevicePtr() const { return m_sphereIds.typedDevicePtr(); }
    const float3* getSphereCentersDevicePtr() const { return m_centers.typedDevicePtr(); }
    const float*  getSphereRadiiDevicePtr() const { return m_radii.typedDevicePtr(); }

  private:
    otk::SyncVector<uint_t>        m_sphereIds;
    otk::SyncVector<float3>        m_centers;
    otk::SyncVector<float>         m_radii;
    otk::SyncVector<std::uint32_t> m_sbtIndices;
    otk::DeviceBuffer              m_devTempBufferGas;
    otk::DeviceBuffer              m_devGeomAs;
};

}  // namespace demandGeometryViewer
