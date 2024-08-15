// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "TestDeviceMemoryPools.h"

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Memory/DeviceRingBuffer.h>
#include <OptiXToolkit/Memory/DeviceFixedPool.h>

#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <fstream>


using namespace otk;

class TestDeviceMemoryPools : public testing::Test
{
  public:
    void SetUp() override
    {
        // Initialize CUDA.
        OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
        OTK_ERROR_CHECK( cudaFree( nullptr ) );
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
        ASSERT_TRUE( pInt == 0 || pInt >= baseInt );
        ASSERT_TRUE( pInt == 0 || pInt < baseInt + size );
    }
}


TEST_F( TestDeviceMemoryPools, TestRingBuffer )
{
    // Create output buffer
    char** devOutput;
    unsigned int outputSize = 1024 * 1024;
    OTK_ERROR_CHECK( cudaMalloc( &devOutput, outputSize * sizeof(char*) ) );

    // Create DeviceRingBuffer allocate by thread, and launch test kernel
    DeviceRingBuffer ringBuffer;
    ringBuffer.init( 32 * 65536, AllocMode::THREAD_BASED );
    ringBuffer.clear( 0 );
    launchDeviceRingBufferTest( ringBuffer, devOutput, outputSize );

    // Copy output buffer back to host, and do a range check on the buffer elements
    std::vector<char*> hostOutput( outputSize );
    OTK_ERROR_CHECK( cudaMemcpy( hostOutput.data(), devOutput,  outputSize * sizeof(char*), cudaMemcpyDeviceToHost ) );
    checkRange( hostOutput, ringBuffer.buffer, ringBuffer.buffSize );
    ringBuffer.tearDown();

    // Create DeviceRingBuffer allocate by warp, and launch test kernel
    ringBuffer.init( 32 * 65536, AllocMode::WARP_NON_INTERLEAVED );
    ringBuffer.clear( 0 );
    launchDeviceRingBufferTest( ringBuffer, devOutput, outputSize );
    
    // Copy output buffer back to host, and do a range check on the buffer elements
    OTK_ERROR_CHECK( cudaMemcpy( hostOutput.data(), devOutput,  outputSize * sizeof(char*), cudaMemcpyDeviceToHost ) );
    checkRange( hostOutput, ringBuffer.buffer, ringBuffer.buffSize );
    ringBuffer.tearDown();

    // Destroy the device-side output buffer
    OTK_ERROR_CHECK( cudaFree( reinterpret_cast<char*>( devOutput ) ) );
}


// Undiagnosed failure: Value of: pInt == 0 || pInt >= baseInt, Actual: false, Expected: true
TEST_F( TestDeviceMemoryPools, DISABLED_TestFixedPool )
{
    // Create output buffer
    char** devOutput;
    unsigned int outputSize = 1024 * 1024;
    OTK_ERROR_CHECK( cudaMalloc( &devOutput, outputSize * sizeof(char*) ) );

    // Create DeviceFixedPool allocate by thread, and launch test kernel
    DeviceFixedPool fixedPool;
    fixedPool.init( 32, 65536, AllocMode::THREAD_BASED );
    fixedPool.clear( 0 );
    launchDeviceFixedPoolTest( fixedPool, devOutput, outputSize );

    // Copy output buffer back to host, and do a range check on the buffer elements
    std::vector<char*> hostOutput( outputSize );
    OTK_ERROR_CHECK( cudaMemcpy( hostOutput.data(), devOutput,  outputSize * sizeof(char*), cudaMemcpyDeviceToHost ) );
    checkRange( hostOutput, fixedPool.buffer, fixedPool.numItemGroups * fixedPool.itemSize );
    fixedPool.tearDown();

    // Create DeviceFixedPool allocate by warp, and launch test kernel
    fixedPool.init( 32, 65536, AllocMode::WARP_NON_INTERLEAVED );
    fixedPool.clear( 0 );
    launchDeviceFixedPoolTest( fixedPool, devOutput, outputSize );
    // Copy output buffer back to host, and do a range check on the buffer elements
    OTK_ERROR_CHECK( cudaMemcpy( hostOutput.data(), devOutput,  outputSize * sizeof(char*), cudaMemcpyDeviceToHost ) );
    checkRange( hostOutput, fixedPool.buffer, fixedPool.numItemGroups * WARP_SIZE * fixedPool.itemSize );
    fixedPool.tearDown();

    // Destroy the device-side output buffer
    OTK_ERROR_CHECK( cudaFree( reinterpret_cast<char*>( devOutput ) ) );
}


TEST_F( TestDeviceMemoryPools, TestInterleavedAccess )
{
    // Create output buffer
    char** devOutput;
    unsigned int outputSize = 65536;
    OTK_ERROR_CHECK( cudaMalloc( &devOutput, outputSize * sizeof(char*) ) );

    // Create DeviceFixedPool allocate by thread, and launch test kernel
    DeviceFixedPool fixedPool;
    fixedPool.init( 32, outputSize, AllocMode::WARP_INTERLEAVED );
    fixedPool.clear( 0 );
    launchDeviceFixedPoolInterleavedTest( fixedPool, devOutput, outputSize );

    // Copy output buffer back to host, and do a range check on the buffer elements
    std::vector<char*> hostOutput( outputSize );
    OTK_ERROR_CHECK( cudaMemcpy( hostOutput.data(), devOutput,  outputSize * sizeof(char*), cudaMemcpyDeviceToHost ) );
    checkRange( hostOutput, fixedPool.buffer, fixedPool.numItemGroups * WARP_SIZE * fixedPool.itemSize );

    // Copy fixed pool memory back, and check it
    int numBytes = fixedPool.numItemGroups * WARP_SIZE * fixedPool.itemSize;
    std::vector<int> hostPoolCopy( numBytes / sizeof(int) );
    OTK_ERROR_CHECK( cudaMemcpy( hostPoolCopy.data(), fixedPool.buffer, numBytes, cudaMemcpyDeviceToHost ) );
    for( unsigned int i=0; i < numBytes / sizeof(int); ++i )
    {
        EXPECT_EQ( hostPoolCopy[i], static_cast<int>( ( i / WARP_SIZE ) % 8 ) );
    }
    fixedPool.tearDown();

    // Destroy the device-side output buffer
    OTK_ERROR_CHECK( cudaFree( reinterpret_cast<char*>( devOutput ) ) );
}
