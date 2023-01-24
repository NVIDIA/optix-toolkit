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

#include <fstream>

#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Memory/DeviceRingBuffer.h"
#include "Memory/DeviceFixedPool.h"
#include "Util/Exception.h"

#include "TestDeviceMemoryPools.h" 

using namespace demandLoading;

class TestDeviceMemoryPools : public testing::Test
{
  public:
    void SetUp() override
    {
        // Initialize CUDA.
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cudaFree( nullptr ) );
    }

  protected:
    const unsigned int m_deviceIndex = 0;
};

void checkRange( std::vector<char*>& addresses, char* ptrBase, unsigned long long size )
{
    for( unsigned int i = 0; i < addresses.size(); ++i )
    {
        unsigned long long pInt = reinterpret_cast<unsigned long long>( addresses[i] );
        unsigned long long baseInt = reinterpret_cast<unsigned long long>( ptrBase );
        EXPECT_TRUE( pInt == 0 || pInt >= baseInt );
        EXPECT_TRUE( pInt == 0 || pInt < baseInt + size );
    }
}


TEST_F( TestDeviceMemoryPools, TestRingBuffer )
{
    // Create output buffer
    char** devOutput;
    unsigned int outputSize = 1024 * 1024;
    DEMAND_CUDA_CHECK( cudaMalloc( &devOutput, outputSize * sizeof(char*) ) );

    // Create DeviceRingBuffer allocate by thread, and launch test kernel
    DeviceRingBuffer ringBuffer;
    ringBuffer.init( 32 * 65536, false );
    ringBuffer.launchPrepare( 0 );
    launchDeviceRingBufferTest( ringBuffer, devOutput, outputSize );

    // Copy output buffer back to host, and do a range check on the buffer elements
    std::vector<char*> hostOutput( outputSize );
    DEMAND_CUDA_CHECK( cudaMemcpy( hostOutput.data(), devOutput,  outputSize * sizeof(char*), cudaMemcpyDeviceToHost ) );
    checkRange( hostOutput, ringBuffer.buffer, ringBuffer.buffSize );
    ringBuffer.tearDown();

    // Create DeviceRingBuffer allocate by warp, and launch test kernel
    ringBuffer.init( 32 * 65536, true );
    ringBuffer.launchPrepare( 0 );
    launchDeviceRingBufferTest( ringBuffer, devOutput, outputSize );
    
    // Copy output buffer back to host, and do a range check on the buffer elements
    DEMAND_CUDA_CHECK( cudaMemcpy( hostOutput.data(), devOutput,  outputSize * sizeof(char*), cudaMemcpyDeviceToHost ) );
    checkRange( hostOutput, ringBuffer.buffer, ringBuffer.buffSize );
    ringBuffer.tearDown();

    // Destroy the device-side output buffer
    DEMAND_CUDA_CHECK( cudaFree( reinterpret_cast<char*>( devOutput ) ) );
}


TEST_F( TestDeviceMemoryPools, TestFixedPool )
{
    // Create output buffer
    char** devOutput;
    unsigned int outputSize = 1024 * 1024;
    DEMAND_CUDA_CHECK( cudaMalloc( &devOutput, outputSize * sizeof(char*) ) );

    // Create DeviceFixedPool allocate by thread, and launch test kernel
    DeviceFixedPool fixedPool;
    fixedPool.init( 32, 65536, false );
    fixedPool.launchPrepare( 0 );
    launchDeviceFixedPoolTest( fixedPool, devOutput, outputSize );

    // Copy output buffer back to host, and do a range check on the buffer elements
    std::vector<char*> hostOutput( outputSize );
    DEMAND_CUDA_CHECK( cudaMemcpy( hostOutput.data(), devOutput,  outputSize * sizeof(char*), cudaMemcpyDeviceToHost ) );
    checkRange( hostOutput, fixedPool.buffer, fixedPool.numItemGroups * fixedPool.itemSize );
    fixedPool.tearDown();

    // Create DeviceFixedPool allocate by warp, and launch test kernel
    fixedPool.init( 32, 65536, true );
    fixedPool.launchPrepare( 0 );
    launchDeviceFixedPoolTest( fixedPool, devOutput, outputSize );
    // Copy output buffer back to host, and do a range check on the buffer elements
    DEMAND_CUDA_CHECK( cudaMemcpy( hostOutput.data(), devOutput,  outputSize * sizeof(char*), cudaMemcpyDeviceToHost ) );
    checkRange( hostOutput, fixedPool.buffer, fixedPool.numItemGroups * WARP_SIZE * fixedPool.itemSize );
    fixedPool.tearDown();

    // Destroy the device-side output buffer
    DEMAND_CUDA_CHECK( cudaFree( reinterpret_cast<char*>( devOutput ) ) );
}
