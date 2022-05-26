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

#include "Util/Exception.h"

#include <condition_variable>
#include <mutex>
#include <vector>

namespace demandLoading {

/// MutexArray is a space-efficient way to emulate a large number of mutexes.  It's used to provide
/// mutual exclusion for thousands of tiles per texture, which is necessary because multiple streams
/// might race to fill a tile, but concurrent memory mapping operations are not permitted by CUDA.
/// The implementation uses a single mutex to guard a bit vector that indicates which items are
/// currently locked.  When a thread attempts to lock an item that is already locked, it is
/// suspended by waiting on a condition variable, which is notified when the item is unlocked.
/// This is a performance compromise, since all threads are woken when any item is unlocked.
class MutexArray
{
  public:
    /// Construct a MutexArray of the specified size.
    MutexArray( unsigned int size )
        : m_excluded( size, false )
    {
    }

    /// Lock the item represented by the specified index.
    void lock( unsigned int index )
    {
        DEMAND_ASSERT( index < m_excluded.size() );
        std::unique_lock<std::mutex> lock( m_mutex );
        m_condition.wait( lock, [this, index] { return !m_excluded[index]; } );
        m_excluded[index] = true;
    }

    /// Unlock the item represented by the specified index.
    void unlock( unsigned int index )
    {
        DEMAND_ASSERT( index < m_excluded.size() );
        {
            std::unique_lock<std::mutex> lock( m_mutex );
            DEMAND_ASSERT( m_excluded[index] );
            m_excluded[index] = false;
        }
        m_condition.notify_all();
    }

    /// Not copyable.
    MutexArray( const MutexArray& ) = delete;

    /// Not assignable.
    MutexArray& operator=( const MutexArray& ) = delete;

  private:
    std::mutex              m_mutex;
    std::condition_variable m_condition;
    std::vector<bool>       m_excluded;
};


/// MutexArrayLock is a scoped lock for a single index in a MutexArray.  It's analogous to
/// std::unique_lock, which can't be used with MutexArray because its lock and unlock methods don't
/// satisfy the BasicLockable requirement (because they require an index argument).
class MutexArrayLock
{
  public:
    /// Lock the given MutexArray at the specified index.
    MutexArrayLock( MutexArray* mutex, unsigned int index )
        : m_mutex( mutex )
        , m_index( index )
    {
        mutex->lock( index );
    }

    /// Unlock the MutexArray wrapped by this lock.
    ~MutexArrayLock() { m_mutex->unlock( m_index ); }

    /// Not copyable.
    MutexArrayLock( MutexArrayLock& ) = delete;

    /// Not assignable.
    MutexArrayLock& operator=( MutexArrayLock& ) = delete;

  private:
    MutexArray*  m_mutex;
    unsigned int m_index;
};

}  // namespace demandLoading
