// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>
#include <OptiXToolkit/Error/ErrorCheck.h>

#include <sstream>

namespace otk {
namespace error {

/// Specialization for CuOmmBaking error names.
template <>
inline std::string getErrorName( cuOmmBaking::Result value )
{
    switch (value)
    {
    case cuOmmBaking::Result::SUCCESS:
        return "SUCCESS";
    case cuOmmBaking::Result::ERROR_CUDA:
        return "ERROR_CUDA";
    case cuOmmBaking::Result::ERROR_INTERNAL:
        return "ERROR_INTERNAL";
    case cuOmmBaking::Result::ERROR_INVALID_VALUE:
        return "ERROR_INVALID_VALUE";
    case cuOmmBaking::Result::ERROR_MISALIGNED_ADDRESS:
        return "ERROR_MISALIGNED_ADDRESS";
    default:
        break;
    }
    return "UNKNOWN";
}

/// Specialization for CuOmmBaking error messages.
template <>
inline std::string getErrorMessage( cuOmmBaking::Result value )
{
    switch (value)
    {
    case cuOmmBaking::Result::SUCCESS:
        return "success";
    case cuOmmBaking::Result::ERROR_CUDA:
        return "CUDA error";
    case cuOmmBaking::Result::ERROR_INTERNAL:
        return "internal error";
    case cuOmmBaking::Result::ERROR_INVALID_VALUE:
        return "invalid value";
    case cuOmmBaking::Result::ERROR_MISALIGNED_ADDRESS:
        return "misaligned address";
    default:
        break;
    }
    return "unknown error";
}

}  // namespace error
}  // namespace otk
