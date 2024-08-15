// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

namespace demandPbrtScene {

struct GeometryCacheStatistics
{
    unsigned int       numTraversables;
    unsigned int       numTriangles;
    unsigned int       numSpheres;
    unsigned int       numNormals;
    unsigned int       numUVs;
    unsigned long long totalBytesRead;
    double             totalReadTime;
};

}  // namespace demandPbrtScene
