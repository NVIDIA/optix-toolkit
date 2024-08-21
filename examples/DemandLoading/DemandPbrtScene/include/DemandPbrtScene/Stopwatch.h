// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <chrono>

namespace demandPbrtScene {

class Stopwatch
{
  public:
    Stopwatch()
        : m_startTime( std::chrono::high_resolution_clock::now() )
    {
    }

    /// Returns the time in seconds since the Stopwatch was constructed.
    double elapsed() const
    {
        using namespace std::chrono;
        return duration_cast<duration<double>>( high_resolution_clock::now() - m_startTime ).count();
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
};

}  // namespace demandPbrtScene
