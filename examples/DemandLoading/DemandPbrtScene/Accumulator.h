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
