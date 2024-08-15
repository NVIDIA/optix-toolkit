// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/Ticket.h>

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <cuda.h>

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace demandLoading {

/// A TicketImpl tracks the progress of a number of tasks.
class TicketImpl
{
  public:
    /// Create a ticket.  Initially the number of tasks is unknown (represented by -1).
    static Ticket create( CUstream stream )
    {
        return Ticket( std::make_shared<TicketImpl>( stream ) );
    }

    /// Get TicketImpl from Ticket, which is held as a shared pointer.
    static std::shared_ptr<TicketImpl>& getImpl( Ticket& ticket ) { return ticket.m_impl; }

    /// Construct TicketImpl with the given stream.
    TicketImpl( CUstream stream )
        : m_stream( stream )
    {
    }

    /// The ticket is updated when the number of tasks are known.
    void update( unsigned int numTasks )
    {
        {
            std::unique_lock<std::mutex> lock( m_mutex );
            m_numTasksTotal     = numTasks;
            m_numTasksRemaining = numTasks;
        }
        // If there are no tasks, notify any threads waiting on the condition variable.
        if( numTasks == 0 )
            m_isDone.notify_all();
    }

    /// Get the stream associated with the ticket.
    CUstream getStream() const { return m_stream; }

    /// Get the total number of tasks tracked by this ticket.  Returns -1 if the number of tasks is
    /// unknown, which indicates that task processing has not yet started.
    int numTasksTotal() const { return m_numTasksTotal; }

    /// Get the number of tasks remaining.  Returns -1 if the number of tasks is unknown, which
    /// indicates that task processing has not yet started.
    int numTasksRemaining() const
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_numTasksRemaining;
    }

    /// Wait for the host-side execution of the tasks to finish.  Optionally, if a CUDA event is
    /// provided, it is recorded when the last task is finished, allowing the caller to wait for
    /// device-side execution to finish (e.g. via cuEventSynchronize or cuStreamWaitEvent).
    void wait( CUevent* event = nullptr )
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        m_isDone.wait( lock, [this] { return m_numTasksRemaining == 0; } );
        if( event )
        {
            OTK_ERROR_CHECK( cuEventRecord( *event, m_stream ) );
        }
    }

    /// Decrement the number of tasks remaining, notifying any waiting threads
    /// when all the tasks are done.
    void notify()
    {
        std::unique_lock<std::mutex> lock( m_mutex );

        // Atomically decrement the number of tasks remaining.
        OTK_ASSERT( m_numTasksRemaining > 0 );
        --m_numTasksRemaining;

        // If there are no tasks remaining, notify any threads waiting on the condition variable.
        // It's not necessary to acquire the mutex.  Redundant notifications are OK.
        if( m_numTasksRemaining == 0 )
            m_isDone.notify_all();
    }

  private:
    const CUstream          m_stream{};
    int                     m_numTasksTotal{-1};
    int                     m_numTasksRemaining{-1};
    mutable std::mutex      m_mutex;
    std::condition_variable m_isDone;
};

}  // namespace demandLoading
