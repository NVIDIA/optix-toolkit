// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

const int BUTTON_SIZE    = 12;
const int BUTTON_SPACING = 5;

struct TexturePaintingParams
{
    int numCanvases;
    int activeCanvas;
    int brushWidth;
    int brushHeight;
    float4 brushColor;
};
