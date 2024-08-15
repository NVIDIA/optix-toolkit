// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cassert>
#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>

#include <OptiXToolkit/CuOmmBaking/CuBuffer.h>
#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>

class OptixOmmArray
{
  public:

    typedef std::vector<OptixOpacityMicromapHistogramEntry> HistogramEntries_t;
    typedef std::vector<OptixOpacityMicromapUsageCount>     UsageCounts_t;

    OptixOmmArray() {};
    ~OptixOmmArray() { destroy(); };

    OptixOmmArray( const OptixOmmArray& source ) = delete;
    OptixOmmArray( OptixOmmArray&& source ) noexcept { swap( source ); }

    OptixOmmArray& operator=( const OptixOmmArray& source ) = delete;

    OptixOmmArray& operator=( OptixOmmArray&& source ) noexcept
    {
        swap( source );
        return *this;
    }

    cuOmmBaking::Result create( OptixDeviceContext context, const cuOmmBaking::BakeOptions& inOptions, const cuOmmBaking::BakeInputDesc* inputs, unsigned int numInputs );

    cuOmmBaking::Result destroy();

    // return the number of omms
    uint32_t getNumOmms() const { return m_info.numMicromapDescs; }

    // return the number of omm uses
    uint32_t getNumOmmUses() const { return m_totalUsageCount; }

    // return the number of build inputs
    uint32_t getNumBuildInputs() const { return (uint32_t)m_optixOmmBuildInputs.size(); }

    // return the omm array histogram
    const HistogramEntries_t& getHistogramEntries() const { return m_histogram; }

    // return the omm array histogram
    const UsageCounts_t& getUsageCounts( uint32_t input ) const { return m_usageCounts[input]; }

    const OptixBuildInputOpacityMicromap& getBuildInput( uint32_t input ) const { return m_optixOmmBuildInputs[input]; }

    // return post build info
    const cuOmmBaking::PostBakeInfo& getPostBakeInfo() const { return m_info; };

  private:
    void swap( OptixOmmArray& source ) noexcept
    {
        std::swap( m_totalUsageCount, source.m_totalUsageCount );
        std::swap( m_histogram, source.m_histogram );
        std::swap( m_usageCounts, source.m_usageCounts );
        std::swap( m_info, source.m_info );
        std::swap( m_optixOmmBuildInputs, source.m_optixOmmBuildInputs );
        std::swap( m_optixOmmArray, source.m_optixOmmArray );
        std::swap( m_optixOmmIndices, source.m_optixOmmIndices );
    }

    uint32_t m_totalUsageCount = {};
    
    cuOmmBaking::PostBakeInfo m_info      = {};

    HistogramEntries_t                          m_histogram = {};
    std::vector<UsageCounts_t>                  m_usageCounts;
    std::vector<OptixBuildInputOpacityMicromap> m_optixOmmBuildInputs;

    CuBuffer<>                                  m_optixOmmArray;
    CuBuffer<>                                  m_optixOmmIndices;
    
};
