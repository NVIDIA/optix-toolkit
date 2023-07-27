//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <OptiXToolkit/Memory/BitCast.h>
#include <OptiXToolkit/Memory/CudaCheck.h>

#include <cuda.h>

#include <stdexcept>

#include <stddef.h>

namespace otk {

/// A simple wrapper around a piece of device memory.
///
/// The lifetime of the memory is the same as the lifetime of this wrapper.
///
class DeviceBuffer
{
  public:
    /// DeviceBuffers are default constructable.
    DeviceBuffer() = default;
    /// DeviceBuffers are constructable from a given size.
    explicit DeviceBuffer( size_t size ) { allocate( size ); }

    ~DeviceBuffer() noexcept
    {
        if( m_devStorage )
            OTK_MEMORY_CUDA_CHECK_NOTHROW( cuMemFree( m_devStorage ) );
    }

    /// DeviceBuffers are not copyable.
    DeviceBuffer( const DeviceBuffer& rhs )            = delete;
    DeviceBuffer& operator=( const DeviceBuffer& rhs ) = delete;

    /// DeviceBuffers are move constructable.
    DeviceBuffer( DeviceBuffer&& rhs ) noexcept
        : m_devStorage( rhs.m_devStorage )
        , m_capacity( rhs.m_capacity )
        , m_size( rhs.m_size )
    {
        rhs.m_devStorage = CUdeviceptr{};
        rhs.m_capacity   = 0;
        rhs.m_size       = 0;
    }

    /// DeviceBuffers are not move assignable (this would potentially leak memory).
    DeviceBuffer operator=( DeviceBuffer&& rhs ) = delete;

    /// Query the size of the allocated memory that is in-use.
    size_t size() const { return m_size; }
    /// Query the capacity of the allocated memory.  May be larger than the size.
    size_t capacity() const { return m_capacity; }

    /// Allocate a chunk of CUDA device memory.
    /// @param size The amount of memory to allocate
    void allocate( size_t size )
    {
        if( m_devStorage != CUdeviceptr{} )
            throw std::runtime_error( "DeviceBuffer already allocated." );
        m_size     = size;
        m_capacity = size;
        OTK_MEMORY_CUDA_CHECK( cuMemAlloc( &m_devStorage, m_capacity ) );
    }

    ///. Free the associated CUDA device memory.
    void free()
    {
        if( m_devStorage != CUdeviceptr{} )
        {
            OTK_MEMORY_CUDA_CHECK( cuMemFree( m_devStorage ) );
            m_devStorage = CUdeviceptr{};
            m_size       = 0;
            m_capacity   = 0;
        }
    }

    /// Resize the associated CUDA device memory, possibly performing a free and allocate.
    /// DeviceBuffers only grow, they never shrink.
    /// @param size The newly requested size.
    void resize( size_t size )
    {
        if( size <= m_capacity )
        {
            m_size = size;
            return;
        }
        free();
        allocate( size );
    }

    /// Return the raw device pointer of the allocated device memory.
    operator CUdeviceptr() const { return m_devStorage; }

    /// Return the associated void* with the allocated device memory.
    void* devicePtr() const { return bit_cast<void*>( m_devStorage ); }

  private:
    CUdeviceptr m_devStorage{};
    size_t      m_capacity{};
    size_t      m_size{};
};

}  // namespace otk
