//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
