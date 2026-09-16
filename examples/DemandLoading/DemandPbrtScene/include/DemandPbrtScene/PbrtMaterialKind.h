// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <string_view>

namespace demandPbrtScene {

enum class PbrtMaterialKind
{
    UNKNOWN,
    MATTE,
    PLASTIC,
    UBER,
    MIRROR,
    GLASS,
    METAL,
    SUBSTRATE,
    TRANSLUCENT,
    SUBSURFACE,
    KD_SUBSURFACE,
    MIX,
    FOURIER,
    HAIR,
    MEASURED,
};

enum class PbrtMaterialCapability : unsigned int
{
    NONE           = 0U,
    GENERATED_MDL  = 1U << 0,
    NAMED_MDL      = 1U << 1,
    KD             = 1U << 2,
    KS             = 1U << 3,
    KR             = 1U << 4,
    KT             = 1U << 5,
    ROUGHNESS      = 1U << 6,
    AXIS_ROUGHNESS = 1U << 7,
};

struct PbrtMaterialDescriptor
{
    PbrtMaterialKind       kind;
    std::string_view       type;
    PbrtMaterialCapability capabilities;

    bool has( PbrtMaterialCapability capability ) const;
};

const PbrtMaterialDescriptor& pbrtMaterialDescriptor( std::string_view type );
PbrtMaterialKind              pbrtMaterialKind( std::string_view type );

}  // namespace demandPbrtScene
