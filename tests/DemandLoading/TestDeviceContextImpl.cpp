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

#include "DeviceContextImpl.h"

#include <gtest/gtest.h>

using namespace demandLoading;

class TestDeviceContextImpl : public testing::Test
{
  public:
    const unsigned int m_deviceIndex = 0;
    Options            m_options{};
    TestDeviceContextImpl()
    {
        m_options.numPages          = 1025;
        m_options.maxRequestedPages = 65;
        m_options.maxFilledPages    = 63;
        m_options.maxStalePages     = 33;
        m_options.maxEvictablePages = 31;
        m_options.maxEvictablePages = 17;
        m_options.useLruTable       = true;
    }
};

TEST_F( TestDeviceContextImpl, TestConstructor )
{
    // Alignment is checked by assertions in the constructor.
    BulkDeviceMemory  memory( m_deviceIndex );
    DeviceContextImpl context{};

    context.reservePerDeviceData( &memory, m_options );
    context.reservePerStreamData( &memory, m_options );
    context.allocatePerDeviceData( &memory, m_options );
    context.allocatePerStreamData( &memory, m_options );
}
