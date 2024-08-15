// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>

namespace cuOmmBaking {

class Exception : public std::runtime_error
{
  public:
    explicit Exception( Result result, const char* msg )
        : std::runtime_error( msg )
        , m_result( result )
    {
    }

    explicit Exception( Result result, const std::string& msg )
        : std::runtime_error( msg )
        , m_result( result )
    {
    }

    Result getResult() const
    {
        return m_result;
    }

  private:
    Result m_result = Result::SUCCESS;
};


//------------------------------------------------------------------------------
//
// Assertions
//
//------------------------------------------------------------------------------

#define OMM_ASSERT( cond )                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        if( !( cond ) )                                                                                                \
        {                                                                                                              \
            std::stringstream ss;                                                                                      \
            ss << __FILE__ << " (" << __LINE__ << "): " << #cond;                                                      \
            throw Exception( ss.str().c_str() );                                                                       \
        }                                                                                                              \
    } while( 0 )


#define OMM_ASSERT_MSG( cond, msg )                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if( !( cond ) )                                                                                                \
        {                                                                                                              \
            std::stringstream ss;                                                                                      \
            ss << ( msg ) << ": " << __FILE__ << " (" << __LINE__ << "): " << #cond;                                   \
            throw Exception( ss.str().c_str() );                                                                       \
        }                                                                                                              \
    } while( 0 )


//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

inline void checkCudaError( cudaError_t error, const char* expr, const char* /*file*/, unsigned int /*line*/ )
{
    if( error != cudaSuccess )
    {
        std::stringstream ss;
        ss << "CUDA call (" << expr << " ) failed with error: '" << cudaGetErrorString( error ) << "' (" __FILE__ << ":"
           << __LINE__ << ")\n";
        throw Exception( Result::ERROR_CUDA, ss.str().c_str() );
    }
}

// A non-throwing variant for use in destructors.
inline void checkCudaErrorNoThrow( cudaError_t error, const char* expr, const char* file, unsigned int line ) noexcept
{
    if( error != cudaSuccess )
    {
        std::cerr << "CUDA call (" << expr << " ) failed with error: '" << cudaGetErrorString( error ) << "' (" << file
                  << ":" << line << ")\n";
#ifndef NDEBUG
        std::terminate();
#endif        
    }
}

inline void checkCudaError( CUresult result, const char* expr, const char* file, unsigned int line )
{
    if( result != CUDA_SUCCESS )
    {
        const char* errorStr;
        cuGetErrorString( result, &errorStr );
        std::stringstream ss;
        ss << "CUDA call (" << expr << " ) failed with error: '" << errorStr << "' (" << file << ":" << line << ")\n";
        throw Exception( Result::ERROR_CUDA, ss.str().c_str() );
    }
}

// A non-throwing variant for use in destructors.
inline void checkCudaErrorNoThrow( CUresult result, const char* expr, const char* file, unsigned int line ) noexcept
{
    if( result != CUDA_SUCCESS )
    {
        const char* errorStr;
        cuGetErrorString( result, &errorStr );
        std::cerr << "CUDA call (" << expr << " ) failed with error: '" << errorStr << "' (" << file << ":" << line << ")\n";
#ifndef NDEBUG
        std::terminate();
#endif
    }
}

#ifndef NDEBUG

#define OMM_CUDA_CHECK( call )                                                              \
    do {                                                                                    \
        cuOmmBaking::checkCudaError( call, #call, __FILE__, __LINE__ );                       \
        cuOmmBaking::checkCudaError( cudaDeviceSynchronize() , #call, __FILE__, __LINE__ );   \
    } while(false);

#define OMM_CUDA_CHECK_NOTHROW( call )                                                      \
    do {                                                                                    \
        cuOmmBaking::checkCudaErrorNoThrow( call, #call, __FILE__, __LINE__ );                \
        cuOmmBaking::checkCudaError( cudaDeviceSynchronize() , #call, __FILE__, __LINE__ );   \
    } while(false);

#else

#define OMM_CUDA_CHECK( call ) cuOmmBaking::checkCudaError( call, #call, __FILE__, __LINE__ )
#define OMM_CUDA_CHECK_NOTHROW( call ) cuOmmBaking::checkCudaErrorNoThrow( call, #call, __FILE__, __LINE__ )

#endif

}  // end namespace cuOmmBaking
