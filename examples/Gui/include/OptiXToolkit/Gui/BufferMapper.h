// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>

namespace otk {

/// Map an otk::CUDAOutputBuffer<T> in an RAII fashion.
///
/// The constructor calls map() and the destructor calls unmap().
///
template <typename BufferElement>
class BufferMapper
{
  public:
    /// Constructor
    ///
    /// @param buffer The buffer to map and unmap.
    ///
    BufferMapper( CUDAOutputBuffer<BufferElement>& buffer )
        : m_buffer( buffer )
        , m_ptr( buffer.map() )
    {
    }
    ~BufferMapper() { m_buffer.unmap(); }

    /// Conversion operators to pointer to buffer element type.
    operator BufferElement*() { return m_ptr; }
    operator const BufferElement*() const { return m_ptr; }

  private:
    CUDAOutputBuffer<BufferElement>& m_buffer;
    BufferElement*                   m_ptr;
};

}  // namespace otk
