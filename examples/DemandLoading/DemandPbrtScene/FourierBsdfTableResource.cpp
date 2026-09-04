// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/FourierBsdfTableResource.h"

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <cuda.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace demandPbrtScene {
namespace {

uint_t checkedUInt( std::size_t value, const char* name )
{
    if( value > std::numeric_limits<uint_t>::max() )
    {
        throw std::runtime_error( std::string{ "Fourier BSDF table " } + name + " exceeds uint_t range" );
    }
    return static_cast<uint_t>( value );
}

template <typename T>
CUdeviceptr uploadVector( const std::vector<T>& values, otk::DeviceBuffer& buffer )
{
    if( values.empty() )
    {
        buffer.free();
        return CUdeviceptr{};
    }

    const std::size_t byteCount{ values.size() * sizeof( T ) };
    buffer.resize( byteCount );
    OTK_ERROR_CHECK( cuMemcpyHtoD( buffer, values.data(), byteCount ) );
    return buffer;
}

}  // namespace

FourierBsdfTableDeviceData makeFourierBsdfTableDeviceData( const FourierBsdfTable& table,
                                                           CUdeviceptr             mu,
                                                           CUdeviceptr             cdf,
                                                           CUdeviceptr             coefficientOffsets,
                                                           CUdeviceptr             coefficientCounts,
                                                           CUdeviceptr             zeroOrderCoefficients,
                                                           CUdeviceptr             coefficients )
{
    FourierBsdfTableDeviceData result{};
    result.flags                 = table.flags;
    result.nMu                   = table.nMu;
    result.nCoefficients         = table.nCoefficients;
    result.maxOrder              = table.maxOrder;
    result.nChannels             = table.nChannels;
    result.nBases                = table.nBases;
    result.eta                   = table.eta;
    result.trailingByteCount     = checkedUInt( table.trailingByteCount, "trailing byte count" );
    result.gridSize              = checkedUInt( table.coefficientOffsets.size(), "grid size" );
    result.mu                    = mu;
    result.cdf                   = cdf;
    result.coefficientOffsets    = coefficientOffsets;
    result.coefficientCounts     = coefficientCounts;
    result.zeroOrderCoefficients = zeroOrderCoefficients;
    result.coefficients          = coefficients;
    return result;
}

void FourierBsdfTableDeviceResource::upload( const FourierBsdfTable& table )
{
    const CUdeviceptr mu{ uploadVector( table.mu, m_mu ) };
    const CUdeviceptr cdf{ uploadVector( table.cdf, m_cdf ) };
    const CUdeviceptr coefficientOffsets{ uploadVector( table.coefficientOffsets, m_coefficientOffsets ) };
    const CUdeviceptr coefficientCounts{ uploadVector( table.coefficientCounts, m_coefficientCounts ) };
    const CUdeviceptr zeroOrderCoefficients{ uploadVector( table.zeroOrderCoefficients, m_zeroOrderCoefficients ) };
    const CUdeviceptr coefficients{ uploadVector( table.coefficients, m_coefficients ) };

    m_hostData = makeFourierBsdfTableDeviceData( table, mu, cdf, coefficientOffsets, coefficientCounts,
                                                 zeroOrderCoefficients, coefficients );
    m_deviceData.resize( sizeof( m_hostData ) );
    OTK_ERROR_CHECK( cuMemcpyHtoD( m_deviceData, &m_hostData, sizeof( m_hostData ) ) );
}

}  // namespace demandPbrtScene
