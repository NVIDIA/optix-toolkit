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

#include <cassert>
#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>

#include <OptiXToolkit/OmmBaking/OmmBaking.h>

#include "CuBuffer.h"

class OptixOmmArray
{
  public:

    typedef std::vector<OptixOpacityMicromapHistogramEntry> ArrayHistogram_t;
    typedef std::vector<OptixOpacityMicromapUsageCount>     IndexHistogram_t;

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

    ommBaking::Result create( OptixDeviceContext context, const ommBaking::Options& inOptions, const ommBaking::BakeInputDesc* inputs, unsigned int numInputs );

    ommBaking::Result destroy();

    // return the number of omms
    uint32_t getNumOmms() const { return m_info.numOmmsInArray; }

    // return the number of omm uses
    uint32_t getNumOmmUses() const { return m_usageCount; }

    // return the number of build inputs
    uint32_t getNumBuildInputs() const { return (uint32_t)m_optixOmmBuildInputs.size(); }

    // return the omm array histogram
    const ArrayHistogram_t& getHistogram() const { return m_histogram; }

    // return the omm array histogram
    const IndexHistogram_t& getUsageHistogram( uint32_t input ) const { return m_usage[input]; }

    const OptixBuildInputOpacityMicromap& getBuildInput( uint32_t input ) const { return m_optixOmmBuildInputs[input]; }

    // return post build info
    const ommBaking::BakePostBuildInfo& getInfo() const { return m_info; };

  private:
    void swap( OptixOmmArray& source ) noexcept
    {
        std::swap( m_usageCount, source.m_usageCount );
        std::swap( m_histogram, source.m_histogram );
        std::swap( m_usage, source.m_usage );
        std::swap( m_info, source.m_info );
        std::swap( m_optixOmmBuildInputs, source.m_optixOmmBuildInputs );
        std::swap( m_optixOmmArray, source.m_optixOmmArray );
        std::swap( m_optixOmmIndices, source.m_optixOmmIndices );
    }

    uint32_t m_usageCount = {};
    
    ommBaking::BakePostBuildInfo m_info      = {};

    ArrayHistogram_t                            m_histogram = {};
    std::vector<IndexHistogram_t>               m_usage;
    std::vector<OptixBuildInputOpacityMicromap> m_optixOmmBuildInputs;

    CuBuffer<>                                  m_optixOmmArray;
    CuBuffer<>                                  m_optixOmmIndices;
    
};
