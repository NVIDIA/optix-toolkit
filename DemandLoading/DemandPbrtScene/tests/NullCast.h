//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

/// null_cast<>
///
/// Because gtest is heavily reliant on macros, sometimes a strongly
/// typed pointer will be compared against nullptr_t (the type of nullptr)
/// and this results in a compilation error.  This is kinda dumb since
/// nullptr is guaranteed to be implicitly convertable to any pointer
/// type and should probably be fixed in gtest, but this simple inline
/// function solves the problem.  Used in scenarios like:
///
/// EXPECT_THAT( devicePointer, hasDeviceMatcherWithArgument( null_cast<TriangleNormal>() ) );
///
/// The problem occurs in this scenario because matchers are declared as
/// template classes that deduce the type of their arguments from the actual
/// parameters and the deduced type of nullptr is nullptr_t.
///
/// null_cast<> isn't needed for simple value comparisons like EXPECT_EQ.
///
/// @tparam T   The desired type that should be pointed to by nullptr.
///             Since nullptr is always a pointer type, it is not necessary
///             to specify T as a pointer type argument as shown above.
///
template <typename T>
T* null_cast()
{
    return nullptr;
}
