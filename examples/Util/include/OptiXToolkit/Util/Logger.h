// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <optix.h>

namespace otk {
namespace util {

/// Sets the log callback to a simple logger that logs to cerr in a
/// standard format used by the example programs.  The log level is
/// set to 4.
///
/// @param options  The options structure on which the log callback and level will be set.
///
void setLogger( OptixDeviceContextOptions& options );

}  // namespace util
}  // namespace otk
