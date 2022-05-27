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

#include "Memory/DeviceContextPool.h"

#include <gtest/gtest.h>

using namespace demandLoading;

class TestDeviceContextPool : public testing::Test
{
  public:
    const unsigned int m_deviceIndex = 0;
    Options            m_options{};

    TestDeviceContextPool()
    {
        m_options.numPages          = 1025;
        m_options.maxRequestedPages = 65;
        m_options.maxFilledPages    = 63;
        m_options.maxStalePages     = 33;
        m_options.maxEvictablePages = 31;
        m_options.maxEvictablePages = 17;
        m_options.useLruTable       = true;
        m_options.maxActiveStreams  = 2;
    }
};

TEST_F( TestDeviceContextPool, Test )
{
    DeviceContextPool pool( m_deviceIndex, m_options );

    DeviceContext* c1 = pool.allocate();
    DeviceContext* c2 = pool.allocate();
    EXPECT_NE( c1, c2 );

    pool.free( c1 );
    DeviceContext* c1a = pool.allocate();
    EXPECT_EQ( c1, c1a );

    pool.free( c2 );
    DeviceContext* c2a = pool.allocate();
    EXPECT_EQ( c2, c2a );
}
