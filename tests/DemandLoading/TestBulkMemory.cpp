//
//  Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
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
