// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandLoading/DeviceContext.h>

#include <vector_types.h>

struct Params
{
    demandLoading::DeviceContext m_context;
    float4* m_output;
    unsigned int m_textureId;
    unsigned int m_width;
    unsigned int m_height;
};

struct RayGenData
{
    float4 m_nonResidentColor;
};
