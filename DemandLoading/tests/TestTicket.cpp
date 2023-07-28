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

#include "TicketImpl.h"

#include <gtest/gtest.h>

#include <thread>

using namespace demandLoading;

class TestTicket : public testing::Test
{
};

TEST_F( TestTicket, TestNoTasks )
{
    TicketImpl ticket( CUstream{} );
    EXPECT_EQ( -1, ticket.numTasksTotal() );
    EXPECT_EQ( -1, ticket.numTasksRemaining() );

    ticket.update( 0 );
    EXPECT_EQ( 0, ticket.numTasksTotal() );
    EXPECT_EQ( 0, ticket.numTasksRemaining() );
    ticket.wait();
    EXPECT_TRUE( true );
}

TEST_F( TestTicket, TestSingleThreaded )
{
    TicketImpl ticket( CUstream{} );
    ticket.update( 2 );

    EXPECT_EQ( 2, ticket.numTasksTotal() );
    EXPECT_EQ( 2, ticket.numTasksRemaining() );
    ticket.notify();
    EXPECT_EQ( 1, ticket.numTasksRemaining() );
    ticket.notify();
    EXPECT_EQ( 0, ticket.numTasksRemaining() );
    ticket.wait();
    EXPECT_TRUE( true );
}

TEST_F( TestTicket, TestMultiThreaded )
{
    const unsigned int numTasks = 32;
    TicketImpl         ticket( CUstream{} );
    ticket.update( numTasks );
    std::vector<std::thread> workers;

    auto worker = [&ticket]() {
        for( unsigned int i = 0; i < numTasks / 2; ++i )
            ticket.notify();
    };

    workers.emplace_back( worker );
    workers.emplace_back( worker );

    ticket.wait();
    workers[0].join();
    workers[1].join();
}
