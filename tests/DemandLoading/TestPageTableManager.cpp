//
//  Copyright (c) 2019 NVIDIA Corporation.  All rights reserved.
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

#include "PageTableManager.h"

#include <gtest/gtest.h>

using namespace demandLoading;

class DummyRequestHandler : public RequestHandler
{
  public:
    void fillRequest( unsigned int deviceIndex, CUstream stream, unsigned int pageId ) override {}
};

class TestPageTableManager : public testing::Test
{
  public:
    PageTableManager    mgr;
    DummyRequestHandler handler;

    TestPageTableManager()
        : mgr( 1024u * 1024u )
    {
    }
};

TEST_F( TestPageTableManager, TestGetAvailablePages )
{
    unsigned int beginAvailablePages = mgr.getAvailablePages();
    mgr.reserve( 1000u, &handler );
    EXPECT_EQ( 1000u, beginAvailablePages - mgr.getAvailablePages() );
}

TEST_F( TestPageTableManager, TestEmptyNotFound )
{
    EXPECT_EQ( nullptr, mgr.getRequestHandler( 0 ) );
}

TEST_F( TestPageTableManager, TestNotFound )
{
    unsigned int firstPage = mgr.reserve( 1, &handler );

    EXPECT_EQ( nullptr, mgr.getRequestHandler( firstPage + 1 ) );
}

TEST_F( TestPageTableManager, TestFindFirstPage )
{
    unsigned int firstPage = mgr.reserve( 3, &handler );

    EXPECT_EQ( &handler, mgr.getRequestHandler( firstPage ) );
}

TEST_F( TestPageTableManager, TestFindMiddlePage )
{
    unsigned int firstPage = mgr.reserve( 3, &handler );

    EXPECT_EQ( &handler, mgr.getRequestHandler( firstPage + 1 ) );
}

TEST_F( TestPageTableManager, TestFindLastPage )
{
    unsigned int firstPage = mgr.reserve( 3, &handler );

    EXPECT_EQ( &handler, mgr.getRequestHandler( firstPage + 2 ) );
}


TEST_F( TestPageTableManager, TestFindExhaustive )
{
    const unsigned int               count = 100;
    std::vector<unsigned int>        firstPages( count );
    std::vector<DummyRequestHandler> handlers( count );
    for( unsigned int i = 0; i < count; ++i )
    {
        firstPages[i] = mgr.reserve( i+1, &handlers[i] );
    }

    for( unsigned int i = 0; i < count; ++i )
    {
        unsigned int firstPage = firstPages[i];
        unsigned int lastPage  = firstPage + i;
        EXPECT_EQ( &handlers[i], mgr.getRequestHandler( firstPage ) );
        EXPECT_EQ( &handlers[i], mgr.getRequestHandler( lastPage ) );
    }
}
