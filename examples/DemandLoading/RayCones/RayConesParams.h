// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

struct RayConesParams
{
    int minRayDepth;
    int maxRayDepth;
    bool updateRayCones;
    float mipScale;
};
