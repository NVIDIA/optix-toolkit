// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Config.h"

#ifdef OTK_USE_MDL

#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <string>
#include <vector>

namespace demandPbrtScene {

enum class MdlBoundParameterType
{
    COLOR,
    FLOAT,
};

struct MdlBoundMaterialParameter
{
    std::string           name;
    MdlBoundParameterType type{};
    float                 red{};
    float                 green{};
    float                 blue{};
    float                 value{};
};

std::string namedMaterialParameterName( unsigned int index, const std::string& paramName );
std::string namedMaterialType( const otk::pbrt::PbrtNamedMaterial& material );

std::vector<MdlBoundMaterialParameter> makeMdlBoundMaterialParameters( const otk::pbrt::PbrtMaterial& material );

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL

