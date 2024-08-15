// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "TicketImpl.h"

#include <OptiXToolkit/DemandLoading/Ticket.h>


namespace demandLoading {

int Ticket::numTasksTotal() const
{
    return m_impl ? m_impl->numTasksTotal() : 0;
}

int Ticket::numTasksRemaining() const
{
    return m_impl ? m_impl->numTasksRemaining() : 0;
}

void Ticket::wait( CUevent* event )
{
    if( m_impl )
        m_impl->wait( event );
}

} // namespace demandLoading
