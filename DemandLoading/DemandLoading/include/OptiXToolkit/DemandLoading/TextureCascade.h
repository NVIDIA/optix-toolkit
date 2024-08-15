// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

namespace demandLoading {

#define REQUEST_CASCADE true
#define CASCADE_BASE 64u
#define NUM_CASCADES 8u

inline unsigned int getCascadeLevel( unsigned int texWidth, unsigned int texHeight )
{
    for( unsigned int cascadeLevel = 0; cascadeLevel < NUM_CASCADES; ++cascadeLevel )
    {
        if( ( CASCADE_BASE << cascadeLevel ) >= texWidth && ( CASCADE_BASE << cascadeLevel ) >= texHeight )
            return cascadeLevel;
    }
    return NUM_CASCADES - 1;
}

} // namespace demandLoading
