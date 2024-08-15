// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/Memory/SyncVector.h>
#include <OptiXToolkit/OptiXMemory/Record.h>

#include <optix_stubs.h>

namespace otk {

/// A vector of otk::Record<T> elements that can be synchronized to a CUDA device.
///
/// The lifetime of the device memory matches the lifetime of this class.
/// Device memory is allocated on the first request to copy host memory to
/// the device.  The copy can be done synchronously or asynchronously via
/// a stream.
///
/// @tparam T The type associated with otk::Record<T>
///
template <typename T>
class SyncRecord
{
  public:
    using iterator = typename SyncVector<Record<T>>::iterator;

    /// Default constructor
    ///
    SyncRecord() = default;

    /// Constructor
    ///
    /// @param size The number of elements in the vector.
    ///
    SyncRecord( size_t size )
        : m_records( size )
    {
    }

    /// Accessor for the size of the vector.
    size_t size() const { return m_records.size(); }

    /// Resize the number of records.
    void resize( size_t size )
    {
        m_records.resize( size );
    }

    /// Iterators for the host elements.
    iterator begin() { return m_records.begin(); }
    iterator end() { return m_records.end(); }

    /// Unchecked access to the data portion of the otk::Record
    /// @param i The index of the element to access.
    T& operator[]( size_t i ) { return m_records[i].data; }

    /// Pack the headers of all the records for a single program group.
    void packHeader( OptixProgramGroup group )
    {
        for( Record<T>& record : m_records )
        {
            OTK_ERROR_CHECK( optixSbtRecordPackHeader( group, &record ) );
        }
    }

    // Pack the header of an individual record for a program group.
    void packHeader( size_t index, OptixProgramGroup group )
    {
        OTK_ERROR_CHECK( optixSbtRecordPackHeader( group, &m_records[index] ) );
    }

    /// Conversion operator to CUdeviceptr
    ///
    /// OptiX API methods expect the device pointer to the record array
    /// as a CUdeviceptr.  cudaMalloc types the memory as a void*, so
    /// this conversion operator eases some of the syntactic noise when
    /// filling out OptiX data structures.
    operator CUdeviceptr() { return m_records; }

    /// Synchronously copy host data to the device.  See \ref SyncVector::copyToDevice.
    void copyToDevice() { m_records.copyToDevice(); }
    /// Asynchronously copy host data to the device.  See \ref SyncVector::copyToDeviceAsync.
    ///
    /// @param stream The stream on which to issue the copy.
    ///
    void copyToDeviceAsync( CUstream stream ) { m_records.copyToDeviceAsync( stream ); }

  private:
    SyncVector<Record<T>> m_records;
};

}  // namespace otk
