// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "RequestQueue.h"
#include "TicketImpl.h"

#include <gtest/gtest.h>

namespace {

TEST( RequestQueue, flushDiscardsPendingRequests )
{
    demandLoading::RequestQueue queue( 4 );
    demandLoading::Ticket       ticket = demandLoading::TicketImpl::create( CUstream{} );
    const unsigned int          pageIds[]{1, 2, 3};

    queue.push( pageIds, 3, ticket );
    EXPECT_EQ( 3, ticket.numTasksRemaining() );

    queue.flush();

    EXPECT_EQ( 0, ticket.numTasksRemaining() );
}

}  // namespace
