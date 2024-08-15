// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
    unsigned int numTasks = 32;
    TicketImpl         ticket( CUstream{} );
    ticket.update( numTasks );
    std::vector<std::thread> workers;

    auto worker = [&ticket, numTasks] {
        for( unsigned int i = 0; i < numTasks / 2; ++i )
            ticket.notify();
    };

    workers.emplace_back( worker );
    workers.emplace_back( worker );

    ticket.wait();
    workers[0].join();
    workers[1].join();
}
