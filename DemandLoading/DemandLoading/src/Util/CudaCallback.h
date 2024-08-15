// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <cuda.h>

namespace demandLoading {

/// Adapter for cuLaunchHostFunc, which enqueues a host function call on a CUDA stream.  A derived
/// class implements the virtual callback() method and uses member variables to retain state.
/// The callback object is destroyed after the callback method is invoked.
/// For example:
/// \code
///     CudaCallback::enqueue( stream, new MyCallback( thing1, thing2 ) );
/// \endcode
class CudaCallback
{
  public:
    /// The destructor is virtual to ensure that members of the derived class are properly destroyed.
    virtual ~CudaCallback() = default;

    /// The callback method should be implemented by the derived class.
    virtual void callback() = 0;

    /// Enqueue a callback on the given stream.  The CudaCallback object will be destroyed after its
    /// callback() method is invoked.
    static void enqueue( CUstream stream, CudaCallback* callback )
    {
        // cuLaunchHostFunc requires a function that takes a void*, so we use a static method to
        // delegate to the virtual callback method.
        OTK_ERROR_CHECK( cuLaunchHostFunc( stream, &staticCallback, callback ) );
    }

  private:
    // Given a type-erased CudaCallback object, invoke the virtual callback method and then destroy it.
    static void staticCallback( void* arg )
    {
        CudaCallback* callback = static_cast<CudaCallback*>( arg );
        callback->callback();
        delete callback;
    }
};

}  // namespace demandLoading
