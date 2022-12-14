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
