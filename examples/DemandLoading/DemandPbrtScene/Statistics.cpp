// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Statistics.h"
#include "Timer.h"

#include <iostream>

namespace demandPbrtScene {

void Statistics::report() const
{
    std::cout << "Total time (s):    " << totalTime.getSeconds() << '\n'
              << "    Init time (s): " << initTime  << '\n'
              << "Number of frames:  " << numFrames  << '\n';
}

} // namespace demandPbrtScene
