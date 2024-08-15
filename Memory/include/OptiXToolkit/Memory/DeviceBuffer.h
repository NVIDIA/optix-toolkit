// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Memory/BitCast.h>

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
    /// Construct from existing storage and size and take ownership.
    DeviceBuffer( CUdeviceptr storage, size_t size )
        : m_devStorage( storage )
        , m_capacity( size )
        , m_size( size )
    {
    }

    ~DeviceBuffer() noexcept
    {
        release();
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

    /// DeviceBuffers are move assignable.
    DeviceBuffer& operator=( DeviceBuffer&& rhs ) noexcept
    {
        release();
        m_devStorage     = rhs.m_devStorage;
        m_capacity       = rhs.m_capacity;
        m_size           = rhs.m_size;
        rhs.m_devStorage = CUdeviceptr{};
        rhs.m_capacity   = 0;
        rhs.m_size       = 0;
        return *this;
    }

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
        OTK_ERROR_CHECK( cuMemAlloc( &m_devStorage, m_capacity ) );
    }

    ///. Free the associated CUDA device memory.
    void free()
    {
        if( m_devStorage != CUdeviceptr{} )
        {
            OTK_ERROR_CHECK( cuMemFree( m_devStorage ) );
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

    /// Attach raw storage and size to this buffer.
    void attach( CUdeviceptr storage, size_t size )
    {
        free();
        m_devStorage = storage;
        m_capacity   = size;
        m_size       = size;
    }

    /// Detach the raw storage from this buffer.
    CUdeviceptr detach()
    {
        const CUdeviceptr storage = m_devStorage;
        m_devStorage              = CUdeviceptr{};
        m_capacity                = 0;
        m_size                    = 0;
        return storage;
    }

  private:
    void release() noexcept
    {
        if( m_devStorage )
            OTK_ERROR_CHECK_NOTHROW( cuMemFree( m_devStorage ) );
        m_devStorage = CUdeviceptr{};
    }

    CUdeviceptr m_devStorage{};
    size_t      m_capacity{};
    size_t      m_size{};
};

}  // namespace otk
