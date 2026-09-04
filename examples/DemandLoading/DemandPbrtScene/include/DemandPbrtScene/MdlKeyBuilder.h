// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Config.h"

#ifdef OTK_USE_MDL

#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <string>

namespace demandPbrtScene {

struct MdlShaderKey
{
    std::string signature;
};

bool operator==( const MdlShaderKey& lhs, const MdlShaderKey& rhs );
bool operator!=( const MdlShaderKey& lhs, const MdlShaderKey& rhs );
bool operator<( const MdlShaderKey& lhs, const MdlShaderKey& rhs );

std::string  toString( const MdlShaderKey& key );
MdlShaderKey makeMdlShaderKey( const otk::pbrt::PbrtMaterial& material );

struct MdlMaterialInstanceKey
{
    MdlShaderKey sourceKey;
    std::string  signature;
    bool         sourceShapeProgramReusable{};
};

bool operator==( const MdlMaterialInstanceKey& lhs, const MdlMaterialInstanceKey& rhs );
bool operator!=( const MdlMaterialInstanceKey& lhs, const MdlMaterialInstanceKey& rhs );
bool operator<( const MdlMaterialInstanceKey& lhs, const MdlMaterialInstanceKey& rhs );

std::string            toString( const MdlMaterialInstanceKey& key );
MdlMaterialInstanceKey makeMdlMaterialInstanceKey( const otk::pbrt::PbrtMaterial& material );

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL
