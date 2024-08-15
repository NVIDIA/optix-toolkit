// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>

#include <memory>

/// \file Ticket.h
/// A Ticket tracks the progress of a number of tasks.

namespace demandLoading {

/// A Ticket tracks the progress of a number of tasks.
class Ticket
{
  public:
    /// A default-constructed ticket has no tasks.
    Ticket() {}

    /// Get the total number of tasks tracked by this ticket, if known.  Returns -1 if the number of
    /// tasks is unknown, which indicates that task processing has not yet started.
    int numTasksTotal() const;

    /// Get the number of tasks remaining, if known.  Returns -1 if the number of tasks is unknown,
    /// which indicates that task processing has not yet started.
    int numTasksRemaining() const;

    /// Wait for the host-side execution of the tasks to finish.  Optionally, if a CUDA event is
    /// provided, it is recorded when the last task is finished, allowing the caller to wait for
    /// device-side execution to finish (e.g. via cuEventSynchronize or cuStreamWaitEvent).
    void wait( CUevent* event = nullptr );

  private:
    std::shared_ptr<class TicketImpl> m_impl;

    friend class TicketImpl;

    /// Non-default tickets are constructed via TicketImpl::create().
    Ticket( std::shared_ptr<TicketImpl>&& impl )
        : m_impl( std::move( impl ) )
    {
    }
};

}  // namespace demandLoading
