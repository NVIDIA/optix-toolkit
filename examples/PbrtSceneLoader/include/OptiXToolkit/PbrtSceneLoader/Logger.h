// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <string>

namespace otk {
namespace pbrt {

class Logger
{
  public:
    virtual ~Logger() = default;

    virtual void start( const char* programName ) = 0;
    virtual void stop() = 0;

    virtual void info( std::string text, const char* file, int line ) const = 0;
    virtual void warning( std::string text, const char* file, int line ) const = 0;
    virtual void error( std::string text, const char* file, int line ) const   = 0;
};

}  // namespace pbrt
}  // namespace otk
