//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "Memory/DeviceContextPool.h"
#include "Memory/SamplerPool.h"
#include "Memory/TilePool.h"

#include <DemandLoading/Options.h>

namespace demandLoading {

class DemandLoaderImpl;

class DeviceMemoryManager
{
  public:
    DeviceMemoryManager( unsigned int deviceIndex, const Options& options )
        : m_deviceIndex( deviceIndex )
        , m_deviceContextPool( m_deviceIndex, options )
        , m_samplerPool( m_deviceIndex )
        , m_tilePool( m_deviceIndex, options.maxTexMemPerDevice )
    {
    }

    /// Get the DeviceContextPool for this device.
    DeviceContextPool* getDeviceContextPool() { return &m_deviceContextPool; }

    /// Get the SamplerPool for this device.
    SamplerPool* getSamplerPool() { return &m_samplerPool; }

    /// Get the TilePool for this device.
    TilePool* getTilePool() { return &m_tilePool; }
    
    /// Returns the amount of device memory allocated.
    size_t getTotalDeviceMemory() const
    {
        return m_deviceContextPool.getTotalDeviceMemory() + m_samplerPool.getTotalDeviceMemory() + m_tilePool.getTotalDeviceMemory();
    }

  private:
    unsigned int      m_deviceIndex;
    DemandLoaderImpl* m_loader;
    DeviceContextPool m_deviceContextPool;
    SamplerPool       m_samplerPool;
    TilePool          m_tilePool;
};

}  // namespace demandLoading
