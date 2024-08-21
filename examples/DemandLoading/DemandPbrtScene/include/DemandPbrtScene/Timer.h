// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <chrono>
#include <cstdint>

namespace demandPbrtScene {

class Timer
{
  public:
    using Nanoseconds = uint64_t;

    Timer() { reset(); }

    void reset() { m_startTime = std::chrono::high_resolution_clock::now(); }

    /// Returns the number of nanoseconds since the Timer was constructed or reset.
    Nanoseconds getNanoseconds() const
    {
        using namespace std::chrono;
        return duration_cast<nanoseconds>( high_resolution_clock::now() - m_startTime ).count();
    }

    /// Returns the number of seconds since the Timer was constructed or reset.
    double getSeconds() const
    {
        using namespace std::chrono;
        return duration_cast<duration<double>>( nanoseconds( getNanoseconds() ) ).count();
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
};

}  // namespace demandPbrtScene
