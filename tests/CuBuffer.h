//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <cassert>
#include <iostream>
#include <memory>
#include <stdint.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename T = unsigned char>
class CuPitchedBuffer
{
public:

    CuPitchedBuffer( size_t width = 0, size_t height = 0 ) { alloc( width, height ); }
    CuPitchedBuffer( const CuPitchedBuffer& ) /*= delete; */ { assert( false ); }
    CuPitchedBuffer( CuPitchedBuffer&& source ) noexcept
    {
        swap( source );
    }

    ~CuPitchedBuffer() { free(); }

    cudaError_t alloc( size_t width, size_t height )
    {
        free();

        size_t bytesPerRow = width * sizeof(T);
        size_t pitch = 0;

        T* ptr = nullptr;
        if( bytesPerRow && height )
        {
            cudaError_t result = cudaMallocPitch( &ptr, &pitch, bytesPerRow, height );
            if( result != cudaSuccess )
                return result;
        }

        std::swap( m_pitch, pitch );
        std::swap( m_width, width );
        std::swap( m_height, height );
        std::swap( m_ptr, ptr );

        return cudaSuccess;
    }

    cudaError_t upload( const T* data )
    {
        return upload( data, m_width * sizeof( T ) );
    }

    cudaError_t upload( const T* data, size_t sourcePitchInBytes )
    {
        if( cudaError_t result = cudaMemcpy2D( m_ptr, m_pitch, data, sourcePitchInBytes, m_width * sizeof( T ), m_height, cudaMemcpyHostToDevice ) )
            return result;
        return cudaSuccess;
    }

    cudaError_t allocAndUpload( size_t width, size_t height, const T* data )
    {
        if( cudaError_t result = alloc( width, height ) )
            return result;
        return upload( data );
    }

    cudaError_t allocAndUpload( size_t width, size_t height, const T* data, size_t sourcePitchInBytes )
    {
        if( cudaError_t result = alloc( width, height ) )
            return result;
        return upload( data, sourcePitchInBytes );
    }

    cudaError_t free()
    {
        cudaError_t result = cudaFree( m_ptr );

        m_pitch = 0;
        m_width = 0;
        m_height = 0;
        m_ptr = nullptr;
        return result;
    }

    CUdeviceptr get() const { return reinterpret_cast< CUdeviceptr >( m_ptr ); }

    size_t width() const { return m_width; }
    size_t height() const { return m_height; }
    size_t pitch() const { return m_pitch; }

    CuPitchedBuffer& operator=( const CuPitchedBuffer& source ) = delete;
    CuPitchedBuffer& operator=( CuPitchedBuffer&& source ) noexcept
    {
        swap( source );
        return *this;
    }

private:

    void swap( CuPitchedBuffer& source )
    {
        std::swap( m_pitch, source.m_pitch );
        std::swap( m_width, source.m_width );
        std::swap( m_height, source.m_height );
        std::swap( m_ptr, source.m_ptr );
    }

    size_t m_pitch = {};
    size_t m_width = {};
    size_t m_height = {};
    T* m_ptr = {};


};

template <typename T = unsigned char>
class CuBuffer
{
  public:
    CuBuffer( size_t count = 0 ) { alloc( count ); }
    CuBuffer( const CuBuffer& ) /*= delete; */ { assert( false ); }
    CuBuffer( CuBuffer&& source ) noexcept
    {
        std::swap( m_count, source.m_count );
        std::swap( m_allocCount, source.m_allocCount );
        std::swap( m_ptr, source.m_ptr );
    }

    ~CuBuffer() { free(); }
    cudaError_t alloc( size_t count )
    {
        free();

        T*     ptr        = nullptr;
        size_t allocCount = count;
        if( count )
        {
            cudaError_t result = cudaMalloc( &ptr, allocCount * sizeof( T ) );
            if( result != cudaSuccess )
                return result;
        }

        std::swap( m_allocCount, allocCount );
        std::swap( m_count, count );
        std::swap( m_ptr, ptr );

        return cudaSuccess;
    }
    cudaError_t allocIfRequired( size_t count )
    {
        if( count <= m_allocCount )
        {
            m_count = count;
            return cudaSuccess;
        }
        return alloc( count );
    }
    cudaError_t allocAndUpload( size_t count, const T* data )
    {
        if( cudaError_t result = alloc( count ) )
            return result;
        return upload( data );
    }
    cudaError_t allocAndUpload( const std::vector<T>& data ) { return allocAndUpload( data.size(), data.data() ); }
    CUdeviceptr get() const { return reinterpret_cast<CUdeviceptr>( m_ptr ); }
    CUdeviceptr get( size_t index ) const { return reinterpret_cast<CUdeviceptr>( m_ptr + index ); }
    cudaError_t free()
    {
        cudaError_t result = cudaFree( m_ptr );

        m_count      = 0;
        m_allocCount = 0;
        m_ptr        = nullptr;
        return result;
    }
    CUdeviceptr release()
    {
        m_count             = 0;
        m_allocCount        = 0;
        CUdeviceptr current = reinterpret_cast<CUdeviceptr>( m_ptr );
        m_ptr               = nullptr;
        return current;
    }
    cudaError_t upload( const T* data )
    {
        if( cudaError_t result = cudaMemcpy( m_ptr, data, m_count * sizeof( T ), cudaMemcpyHostToDevice ) )
            return result;
        return cudaSuccess;
    }
    cudaError_t download( T* data ) const
    {
        if( cudaError_t result = cudaMemcpy( data, m_ptr, m_count * sizeof( T ), cudaMemcpyDeviceToHost ) )
            return result;
        return cudaSuccess;
    }
    cudaError_t download( std::vector<T>& data ) const
    {
        data.resize( m_count );
        if( cudaError_t result = cudaMemcpy( data.data(), m_ptr, m_count * sizeof( T ), cudaMemcpyDeviceToHost ) )
            return result;
        return cudaSuccess;
    }
    cudaError_t downloadSub( size_t count, size_t offset, T* data ) const
    {
        assert( count + offset <= m_allocCount );
        if( cudaError_t result = cudaMemcpy( data, m_ptr + offset, count * sizeof( T ), cudaMemcpyDeviceToHost ) )
            return result;
        return cudaSuccess;
    }
    cudaError_t copy( CUdeviceptr data )
    {
        if( cudaError_t result = cudaMemcpy( m_ptr, (void*)data, m_count * sizeof( T ), cudaMemcpyDeviceToDevice ) )
            return result;
        return cudaSuccess;
    }
    cudaError_t set( int value = 0 )
    {
        if( cudaError_t result = cudaMemset( m_ptr, value, m_count * sizeof( T ) ) )
            return result;
        return cudaSuccess;
    }

    CuBuffer& operator=( const CuBuffer& source ) = delete;

    CuBuffer& operator=( CuBuffer&& source ) noexcept
    {
        std::swap( m_count, source.m_count );
        std::swap( m_allocCount, source.m_allocCount );
        std::swap( m_ptr, source.m_ptr );
        return *this;
    }
    size_t count() const { return m_count; }
    size_t reservedCount() const { return m_allocCount; }
    size_t byteSize() const { return m_allocCount * sizeof( T ); }

  private:
    size_t m_count      = 0;
    size_t m_allocCount = 0;
    T*     m_ptr        = nullptr;
};

template <typename T>
std::ostream& operator<<( std::ostream& os, const CuBuffer<T>& buffer )
{
    std::vector<T> h( buffer.count() );
    if( buffer.download( h.data() ) )
        throw std::runtime_error( "cuda error" );

    for( size_t i = 0; i < h.size(); i++ )
    {
        os << "\t" << i << ": " << h[i] << std::endl;
    }

    return os;
}
