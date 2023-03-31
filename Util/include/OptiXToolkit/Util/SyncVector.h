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

#include <OptiXToolkit/Util/Exception.h>

#include <cuda_runtime.h>

#include <vector>

namespace otk {

/// A vector of elements of type T that can be synchronized to a CUDA device.
///
/// The lifetime of the device memory matches the lifetime of this class.
/// Device memory is allocated on the first request to copy host memory to
/// the device.  The copy can be done synchronously or asynchronously via
/// a stream.
///
/// @tparam T The type of the elements.
///
template <typename T>
class SyncVector
{
  public:
    using iterator = typename std::vector<T>::iterator;

    /// Default Constructor
    SyncVector<T>()
        : m_device( nullptr )
    {
    }

    /// Constructor
    ///
    /// @param size The number of elements in the vector.
    ///
    SyncVector<T>( size_t size )
        : m_host( size )
        , m_device( nullptr )
    {
    }
    ~SyncVector<T>() { CUDA_CHECK_NOTHROW( cudaFree( m_device ) ); }

    /// Return whether or not the container is empty.
    bool empty() const { return m_host.empty(); }
    /// Return the number of elements in the vector.
    size_t size() const { return m_host.size(); }
    /// Return the capacity of the vector in elements.
    size_t capacity() const { return m_host.capacity(); }

    /// Unchecked element access.
    T&       operator[]( size_t i ) { return m_host[i]; }
    const T& operator[]( size_t i ) const { return m_host[i]; }
    /// Checked element access.
    T&       at( size_t i ) { return m_host.at( i ); }
    const T& at( size_t i ) const { return m_host.at( i ); }
    // Access the last element.
    T&       back() { return m_host.back(); }
    const T& back() const { return m_host.back(); }

    /// Iterators for the host elements.
    iterator begin() { return m_host.begin(); }
    iterator end() { return m_host.end(); }

    /// Append to the end.
    void push_back( const T& value ) { m_host.push_back( value ); }
    void push_back( T&& value ) { m_host.emplace_back( std::move( value ) ); }

    /// Remove the element at the given position
    void erase( iterator pos ) { m_host.erase( pos ); }

    /// Synchronously copy host data to the device.
    ///
    /// Device memory is allocated on the first copy request.
    ///
    void copyToDevice()
    {
        ensureDeviceMemory();
        CUDA_CHECK( cudaMemcpy( m_device, m_host.data(), m_host.size() * sizeof( T ), cudaMemcpyHostToDevice ) );
    }
    /// Asynchronously copy host data to the device.
    ///
    /// Device memory is allocated synchronously on the first copy request.
    ///
    /// @param stream The stream on which to issue the copy.
    ///
    void copyToDeviceAsync( CUstream stream )
    {
        ensureDeviceMemory();
        CUDA_CHECK( cudaMemcpyAsync( m_device, m_host.data(), m_host.size() * sizeof( T ), cudaMemcpyHostToDevice, stream ) );
    }

    /// Typed pointer to the device memory.
    ///
    /// This method returns nullptr if the data has not been copied to the device.
    T*       typedDevicePtr() { return static_cast<T*>( m_device ); }
    const T* typedDevicePtr() const { return static_cast<const T*>( m_device ); }

    /// Untyped pointer to the device memory.
    ///
    /// This method returns nullptr if the data has not been copied to the device.
    void* devicePtr() { return m_device; }

    /// Conversion operator to CUdeviceptr
    ///
    /// OptiX API methods expect the device pointer to the record array
    /// as a CUdeviceptr.  cudaMalloc types the memory as a void*, so
    /// this conversion operator eases some of the syntactic noise when
    /// filling out OptiX data structures.
    operator CUdeviceptr() { return reinterpret_cast<CUdeviceptr>( m_device ); }

    void resize( size_t size )
    {
        if( size > m_host.capacity() )
        {
            if( m_device )
            {
                CUDA_CHECK( cudaFree( m_device ) );
                m_device = nullptr;
            }
        }
        m_host.resize( size );
    }

    /// Clear the host vector.
    void clear() { m_host.clear(); }

  private:
    void ensureDeviceMemory()
    {
        if( m_device == nullptr )
        {
            const size_t numBytes = m_host.capacity() * sizeof( T );
            CUDA_CHECK( cudaMalloc( &m_device, numBytes ) );
        }
    }

    std::vector<T> m_host;
    void*          m_device;
};

}  // namespace otk
