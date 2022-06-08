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

#include "Memory/Allocators.h"
#include "Util/Exception.h"
#include "Util/Math.h"

namespace demandLoading {

/// BulkMemory encapsulates logic for dividing a single block of memory into multiple regions
/// (with correct alignment).
template <typename Allocator>
class BulkMemory
{
  public:
    /// Construct BulkMemory with given instance of Allocator.
    BulkMemory( const Allocator& allocator )
        : m_allocator( allocator )
    {
    }

    /// Not copyable.
    BulkMemory( const BulkMemory& ) = delete;

    /// Not assignable
    BulkMemory& operator=( const BulkMemory& ) = delete;

    /// Destroy the bulk allocation, freeing the allocated memory.
    ~BulkMemory() { reset(); }

    /// Get the capacity in bytes.
    size_t capacity() const { return m_capacity; }

    /// Reserve storage for an allocation of the specified size and alignment.
    void reserveBytes( size_t numBytes, size_t alignment )
    {
        DEMAND_ASSERT_MSG( m_data == nullptr, "BulkMemory::reserve() cannot be called after allocate()" );
        if( numBytes > 0 )
        {
            m_capacity = align( m_capacity, alignment );
            m_capacity += numBytes;
        }
    }

    /// Reserve storage for an array of the specified type and length.
    template <typename T>
    void reserve( size_t length = 1 )
    {
        reserveBytes( length * sizeof( T ), alignof( T ) );
    }

    /// Allocate memory (if necessary) and return a chunk of the specified size and alignment.
    /// Returns a null pointer if the size is zero.  A sequence of allocate() calls must be preceded
    /// by a matching sequence of reseve() calls.
    template <typename T>
    T* allocateBytes( size_t numBytes, size_t alignment )
    {
        if( numBytes == 0 )
            return nullptr;

        // Allocate reserved memory on first call to allocate().
        if( m_data == nullptr )
        {
            m_data = reinterpret_cast<uint8_t*>( m_allocator.allocate( m_capacity ) );
        }
        m_end = align( m_end, alignment );
        DEMAND_ASSERT( m_end + numBytes <= m_capacity );
        T* result = reinterpret_cast<T*>( m_data + m_end );
        m_end += numBytes;
        return result;
    }

    /// Allocate memory (if necessary) and return an array of the specified length and alignment.
    /// Returns a null pointer if the length is zero.  A sequence of allocate() calls must be
    /// preceded by a matching sequence of reseve() calls.
    template <typename T>
    T* allocate( size_t length = 1 )
    {
        return this->allocateBytes<T>( length * sizeof( T ), alignof( T ) );
    }

    void reset()
    {
        if( m_data != nullptr )
        {
            m_allocator.free( m_data );
            m_data     = nullptr;
            m_capacity = 0;
            m_end      = 0;
        }
    }

    /// Zero-initialize the allocated memory.
    void setToZero()
    {
        if( m_data )
            m_allocator.setToZero( m_data, m_capacity );
    }

  private:
    Allocator m_allocator;
    uint8_t*  m_data     = nullptr;
    size_t    m_capacity = 0;
    size_t    m_end      = 0;
};

class BulkDeviceMemory : public BulkMemory<DeviceAllocator>
{
  public:
    BulkDeviceMemory( unsigned int deviceIndex )
        : BulkMemory<DeviceAllocator>( DeviceAllocator( deviceIndex ) )
    {
    }
};


class BulkPinnedMemory : public BulkMemory<PinnedAllocator>
{
  public:
    BulkPinnedMemory()
        : BulkMemory<PinnedAllocator>( PinnedAllocator() )
    {
    }
};

}  // namespace demandLoading
