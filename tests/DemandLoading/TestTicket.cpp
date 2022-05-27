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

#include "TicketImpl.h"

#include <gtest/gtest.h>

#include <thread>

using namespace demandLoading;

class TestTicket : public testing::Test
{
};

TEST_F( TestTicket, TestNoTasks )
{
    TicketImpl ticket( 0, CUstream() );
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
    TicketImpl ticket( 0, CUstream() );
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
    TicketImpl         ticket( 0, CUstream() );
    ticket.update( numTasks );
    std::vector<std::thread> workers;

    auto worker = [&ticket, numTasks]() {
        for( unsigned int i = 0; i < numTasks / 2; ++i )
            ticket.notify();
    };

    workers.emplace_back( worker );
    workers.emplace_back( worker );

    ticket.wait();
    workers[0].join();
    workers[1].join();
}
