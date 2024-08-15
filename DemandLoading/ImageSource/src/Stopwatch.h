// SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <chrono>

namespace imageSource {

class Stopwatch
{
  public:
    Stopwatch()
        : startTime( std::chrono::high_resolution_clock::now() )
    {
    }

    /// Returns the time in seconds since the Stopwatch was constructed.
    double elapsed() const
    {
        using namespace std::chrono;
        return duration_cast<duration<double>>( high_resolution_clock::now() - startTime ).count();
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

}  // end namespace imageSource
