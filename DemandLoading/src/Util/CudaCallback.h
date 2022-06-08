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
    /// The callback method should be implemented by the derived class.
    virtual void callback() = 0;

    /// The destructor is virtual to ensure that members of the derived class are properly destroyed.
    virtual ~CudaCallback() { }

    /// Enqueue a callback on the given stream.  The CudaCallback object will be destroyed after its
    /// callback() method is invoked.
    static void enqueue( CUstream stream, CudaCallback* callback )
    {
        // cuLaunchHostFunc requires a function that takes a void*, so we use a static method to
        // delegate to the virtual callback method.
        DEMAND_CUDA_CHECK( cuLaunchHostFunc( stream, &staticCallback, callback ) );
    }

  private:
    // Given a type-erased CudaCallback object, invoke the virtual callback method and then destroy it.
    static void staticCallback( void* arg )
    {
        CudaCallback* callback = reinterpret_cast<CudaCallback*>( arg );
        callback->callback();
        delete callback;
    }
};

}  // namespace demandLoading
