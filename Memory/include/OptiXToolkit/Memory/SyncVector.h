// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Memory/DeviceBuffer.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace otk {

/// A vector of elements of type T that can be synchronized to a CUDA device.
///
/// It is the responsibility of the owner of the SyncVector to copy host
/// data to the device after modification.  No attempt is made to track
/// data modifications on the host and automatically copy to the device.
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
    using const_iterator = typename std::vector<T>::const_iterator;
    using iterator       = typename std::vector<T>::iterator;
    using value_type     = T;

    /// Default Constructor
    SyncVector<T>() = default;

    /// Constructor
    ///
    /// @param size The number of elements in the vector.
    ///
    SyncVector<T>( size_t size )
        : m_host( size )
    {
    }
    ~SyncVector<T>() = default;

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
    iterator       begin() { return m_host.begin(); }
    iterator       end() { return m_host.end(); }
    const_iterator cbegin() const { return m_host.cbegin(); }
    const_iterator cend() const { return m_host.cend(); }

    /// Append to the end.
    void push_back( const T& value ) { m_host.push_back( value ); }
    void push_back( T&& value ) { m_host.emplace_back( std::move( value ) ); }

    /// Remove the element at the given position
    void erase( iterator pos ) { m_host.erase( pos ); }

    /// Insert the element at the given position
    void insert( const_iterator pos, const T& value ) { m_host.insert( pos, value ); }

    /// Synchronously copy host data to the device.
    ///
    /// Device memory is allocated on the first copy request.
    ///
    void copyToDevice()
    {
        ensureDeviceMemory();
        OTK_ERROR_CHECK( cudaMemcpy( m_device.devicePtr(), m_host.data(), m_host.size() * sizeof( T ), cudaMemcpyHostToDevice ) );
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
        OTK_ERROR_CHECK( cudaMemcpyAsync( m_device.devicePtr(), m_host.data(), m_host.size() * sizeof( T ), cudaMemcpyHostToDevice, stream ) );
    }

    /// Untyped pointer to the device memory.
    ///
    /// This method returns nullptr if the data has not been copied to the device.
    void* devicePtr() const { return m_device.devicePtr(); }

    /// Typed pointer to the device memory.
    ///
    /// This method returns nullptr if the data has not been copied to the device.
    T*       typedDevicePtr() { return static_cast<T*>( devicePtr() ); }
    const T* typedDevicePtr() const { return static_cast<const T*>( devicePtr() ); }

    /// Conversion operator to CUdeviceptr
    ///
    /// OptiX API methods expect the device pointer to the record array
    /// as a CUdeviceptr.  cudaMalloc types the memory as a void*, so
    /// this conversion operator eases some of the syntactic noise when
    /// filling out OptiX data structures.
    operator CUdeviceptr() { return m_device; }

    /// Resize the host memory to the given number of elements.
    void resize( size_t size )
    {
        m_host.resize( size );
    }

    /// Clear the host vector.
    void clear() { m_host.clear(); }

    /// Detach the device storage
    CUdeviceptr detach() { return m_device.detach(); }

  private:
    void ensureDeviceMemory()
    {
        const size_t hostBytes = m_host.size() * sizeof( T );
        if( m_device.size() < hostBytes )
        {
            m_device.resize( hostBytes );
        }
    }

    std::vector<T> m_host;
    DeviceBuffer   m_device; // A block of untyped bytes on the device.
};

/// Fill a SyncVector<T> with a value of type U that can be converted to T.
template <typename T, typename U>
void fill( SyncVector<T>& vec, U value )
{
    std::fill( std::begin( vec ), std::end( vec ), static_cast<T>( value ) );
}

}  // namespace otk
