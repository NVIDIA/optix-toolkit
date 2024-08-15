// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "Timer.h"

#include <cstdint>

namespace demandPbrtScene {

class Statistics
{
  public:
    double        initTime{};
    Timer         totalTime;
    std::uint64_t numFrames{};

    void report() const;
};

}  // namespace demandPbrtScene
