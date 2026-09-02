// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/PbrtMaterialKind.h"

namespace demandPbrtScene {
namespace {

constexpr PbrtMaterialCapability operator|( PbrtMaterialCapability lhs, PbrtMaterialCapability rhs )
{
    return static_cast<PbrtMaterialCapability>( static_cast<unsigned int>( lhs ) | static_cast<unsigned int>( rhs ) );
}

constexpr PbrtMaterialDescriptor MATERIALS[] = {
    { PbrtMaterialKind::UNKNOWN, "", PbrtMaterialCapability::NONE },
    { PbrtMaterialKind::MATTE, "matte", PbrtMaterialCapability::GENERATED_MDL | PbrtMaterialCapability::NAMED_MDL
                                                   | PbrtMaterialCapability::KD },
    { PbrtMaterialKind::PLASTIC, "plastic", PbrtMaterialCapability::GENERATED_MDL | PbrtMaterialCapability::NAMED_MDL
                                                       | PbrtMaterialCapability::KD | PbrtMaterialCapability::KS
                                                       | PbrtMaterialCapability::ROUGHNESS },
    { PbrtMaterialKind::UBER, "uber", PbrtMaterialCapability::GENERATED_MDL | PbrtMaterialCapability::NAMED_MDL
                                                 | PbrtMaterialCapability::KD | PbrtMaterialCapability::KS
                                                 | PbrtMaterialCapability::KR | PbrtMaterialCapability::KT
                                                 | PbrtMaterialCapability::ROUGHNESS
                                                 | PbrtMaterialCapability::AXIS_ROUGHNESS },
    { PbrtMaterialKind::MIRROR, "mirror", PbrtMaterialCapability::GENERATED_MDL | PbrtMaterialCapability::NAMED_MDL
                                                     | PbrtMaterialCapability::KR },
    { PbrtMaterialKind::GLASS, "glass", PbrtMaterialCapability::GENERATED_MDL | PbrtMaterialCapability::NAMED_MDL
                                                   | PbrtMaterialCapability::KR | PbrtMaterialCapability::KT },
    { PbrtMaterialKind::METAL, "metal", PbrtMaterialCapability::GENERATED_MDL | PbrtMaterialCapability::NAMED_MDL
                                                   | PbrtMaterialCapability::ROUGHNESS
                                                   | PbrtMaterialCapability::AXIS_ROUGHNESS },
    { PbrtMaterialKind::SUBSTRATE, "substrate", PbrtMaterialCapability::GENERATED_MDL
                                                           | PbrtMaterialCapability::NAMED_MDL
                                                           | PbrtMaterialCapability::KD | PbrtMaterialCapability::KS
                                                           | PbrtMaterialCapability::AXIS_ROUGHNESS },
    { PbrtMaterialKind::TRANSLUCENT, "translucent", PbrtMaterialCapability::GENERATED_MDL
                                                               | PbrtMaterialCapability::NAMED_MDL
                                                               | PbrtMaterialCapability::KD | PbrtMaterialCapability::KS
                                                               | PbrtMaterialCapability::ROUGHNESS },
    { PbrtMaterialKind::SUBSURFACE, "subsurface", PbrtMaterialCapability::GENERATED_MDL },
    { PbrtMaterialKind::KD_SUBSURFACE, "kdsubsurface", PbrtMaterialCapability::GENERATED_MDL
                                                                   | PbrtMaterialCapability::KD },
    { PbrtMaterialKind::MIX, "mix", PbrtMaterialCapability::GENERATED_MDL },
    { PbrtMaterialKind::FOURIER, "fourier", PbrtMaterialCapability::NONE },
    { PbrtMaterialKind::HAIR, "hair", PbrtMaterialCapability::NONE },
    { PbrtMaterialKind::MEASURED, "measured", PbrtMaterialCapability::NONE },
};

}  // namespace

bool PbrtMaterialDescriptor::has( PbrtMaterialCapability capability ) const
{
    return ( static_cast<unsigned int>( capabilities ) & static_cast<unsigned int>( capability ) ) != 0U;
}

const PbrtMaterialDescriptor& pbrtMaterialDescriptor( std::string_view type )
{
    for( const PbrtMaterialDescriptor& material : MATERIALS )
    {
        if( material.type == type )
        {
            return material;
        }
    }
    return MATERIALS[0];
}

PbrtMaterialKind pbrtMaterialKind( std::string_view type )
{
    return pbrtMaterialDescriptor( type ).kind;
}

}  // namespace demandPbrtScene
