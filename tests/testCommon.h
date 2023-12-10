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

#include <cuda_runtime.h>

#include "Util/Image.h"
#include "Util/OptiXOmmArray.h"

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>

#include <gtest/gtest.h>

#define OPTIX_THROW( x )                                                                                                                                                                                                                                                               \
    {                                                                                                                                                                                                                                                                                  \
        OptixResult err = (OptixResult)( x );                                                                                                                                                                                                                                          \
        ASSERT_TRUE( err == OPTIX_SUCCESS );                                                                                                                                                                                                                                           \
        if( err )                                                                                                                                                                                                                                                                      \
            throw std::runtime_error( "Optix Error" );                                                                                                                                                                                                                                 \
    };

#define CU_THROW( x )                                                                                                                                                                                                                                                                  \
    {                                                                                                                                                                                                                                                                                  \
        CUresult err = (CUresult)( x );                                                                                                                                                                                                                                                \
        ASSERT_TRUE( err == CUDA_SUCCESS );                                                                                                                                                                                                                                            \
        if( err )                                                                                                                                                                                                                                                                      \
            throw std::runtime_error( "Cuda Driver API Error" );                                                                                                                                                                                                                       \
    };

#define CUDA_THROW( x )                                                                                                                                                                                                                                                                \
    {                                                                                                                                                                                                                                                                                  \
        cudaError_t err = (cudaError_t)( x );                                                                                                                                                                                                                                          \
        ASSERT_TRUE( err == cudaSuccess );                                                                                                                                                                                                                                             \
        if( err )                                                                                                                                                                                                                                                                      \
            throw std::runtime_error( "Cuda Runtime Api Error" );                                                                                                                                                                                                                      \
    };

#define OMM_THROW( x )                                                                                                                                                                                                                                                                 \
    {                                                                                                                                                                                                                                                                                  \
        cuOmmBaking::OmmResult err = ( x );                                                                                                                                                                                                                                                \
        ASSERT_TRUE( err == cuOmmBaking::SUCCESS );                                                                                                                                                                                                                                    \
        if( err )                                                                                                                                                                                                                                                                      \
            throw std::runtime_error( "Ommutil Error" );                                                                                                                                                                                                                               \
    };

#define CU_CHECK( x )                                                                                                                                                                                                                                                                  \
    {                                                                                                                                                                                                                                                                                  \
        CUresult err = (CUresult)( x );                                                                                                                                                                                                                                                \
        if( err )                                                                                                                                                                                                                                                                      \
            return cuOmmBaking::ERROR_CUDA;                                                                                                                                                                                                                                      \
    };

#define OMM_CHECK( x )                                                                                                                                                                                                                                                                 \
    {                                                                                                                                                                                                                                                                                  \
        cuOmmBaking::Result err = ( x );                                                                                                                                                                                                                                                \
        if( err != cuOmmBaking::Result::SUCCESS )                                                                                                                                                                                                                                                                      \
            return err;                                                                                                                                                                                                                                                                \
    };

class TestCommon : public testing::Test
{
  private:
    std::string m_imageNamePrefix;

  protected:
    OptixDeviceContext    optixContext = {};

    void SetUp() override;

    void TearDown() override;

    cuOmmBaking::Result saveImageToFile( std::string imageNamePrefix, const std::vector<uchar3>& image, uint32_t width, uint32_t height );

    void compareImage();
};
