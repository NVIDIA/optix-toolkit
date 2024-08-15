// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
