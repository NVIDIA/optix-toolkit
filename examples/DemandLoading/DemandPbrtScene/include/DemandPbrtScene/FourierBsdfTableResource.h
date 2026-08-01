// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/FourierBsdfTable.h"
#include "DemandPbrtScene/Params.h"

#include <OptiXToolkit/Memory/DeviceBuffer.h>

#include <cuda.h>

namespace demandPbrtScene {

FourierBsdfTableDeviceData makeFourierBsdfTableDeviceData( const FourierBsdfTable& table,
                                                           CUdeviceptr             mu,
                                                           CUdeviceptr             cdf,
                                                           CUdeviceptr             coefficientOffsets,
                                                           CUdeviceptr             coefficientCounts,
                                                           CUdeviceptr             zeroOrderCoefficients,
                                                           CUdeviceptr             coefficients );

class FourierBsdfTableDeviceResource
{
  public:
    void upload( const FourierBsdfTable& table );

    CUdeviceptr deviceData() const { return m_deviceData; }

    const FourierBsdfTableDeviceData& hostData() const { return m_hostData; }

  private:
    FourierBsdfTableDeviceData m_hostData{};
    otk::DeviceBuffer          m_mu;
    otk::DeviceBuffer          m_cdf;
    otk::DeviceBuffer          m_coefficientOffsets;
    otk::DeviceBuffer          m_coefficientCounts;
    otk::DeviceBuffer          m_zeroOrderCoefficients;
    otk::DeviceBuffer          m_coefficients;
    otk::DeviceBuffer          m_deviceData;
};

}  // namespace demandPbrtScene
