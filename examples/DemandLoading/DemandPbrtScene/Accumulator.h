// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>
#include <vector_types.h>

#include <OptiXToolkit/Error/cuErrorCheck.h>

class Accumulator
{
  public:
    void resize( int width, int height )
    {
        if( m_width == width && m_height == height )
            return;
        OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( m_buffer ) ) );
        m_width = width;
        m_height = height;
        m_buffer = nullptr;
        if( m_width * m_height > 0 )
            OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &m_buffer ), m_width * m_height * sizeof( float4 ) ) );
        clear();
    }

    void clear()
    {
        OTK_ERROR_CHECK( cuMemsetD8( reinterpret_cast<CUdeviceptr>( m_buffer ), 0, m_width * m_height * sizeof( float4 ) ) );
    }

    float4* getBuffer() { return m_buffer; }

  private:
    float4* m_buffer = nullptr;
    int m_width = 0;
    int m_height = 0;
};
