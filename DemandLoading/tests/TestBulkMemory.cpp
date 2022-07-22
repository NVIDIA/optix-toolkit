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

#include "Memory/BulkMemory.h"

#include <gtest/gtest.h>

using namespace demandLoading;

static bool isAligned( void* ptr, size_t alignment )
{
    return reinterpret_cast<uintptr_t>( ptr ) % alignment == 0;
}

class TestBulkMemory : public testing::Test
{
};

TEST_F( TestBulkMemory, TestAlignment )
{
    const unsigned int deviceIndex = 0;
    BulkDeviceMemory   memory( deviceIndex );
    memory.reserveBytes( 1, 1 );
    memory.reserveBytes( 2, 2 );
    memory.reserveBytes( 4, 4 );
    memory.reserveBytes( 1, 1 );
    memory.reserveBytes( 8, 8 );
    memory.reserveBytes( 16, 16 );

    EXPECT_TRUE( isAligned( memory.allocateBytes<void*>( 1, 1 ), 1 ) );
    EXPECT_TRUE( isAligned( memory.allocateBytes<void*>( 2, 2 ), 2 ) );
    EXPECT_TRUE( isAligned( memory.allocateBytes<void*>( 4, 4 ), 4 ) );
    EXPECT_TRUE( isAligned( memory.allocateBytes<void*>( 1, 1 ), 1 ) );
    EXPECT_TRUE( isAligned( memory.allocateBytes<void*>( 8, 8 ), 8 ) );
    EXPECT_TRUE( isAligned( memory.allocateBytes<void*>( 16, 16 ), 16 ) );
}
