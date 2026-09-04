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

struct MdlShaderKey;

struct GeneratedMdlSource
{
    std::string              moduleName;
    std::string              materialName;
    std::string              source;
    std::vector<std::string> unsupportedReasons;
};

void appendUnsupportedReason( GeneratedMdlSource& result, const std::string& reason );

GeneratedMdlSource generateMdlSource( const MdlShaderKey& key );
GeneratedMdlSource generateMdlSource( const otk::pbrt::PbrtMaterial& material );

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL

