// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Params.h"

#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <cuda_runtime.h>

namespace demandPbrtScene {

inline MaterialFlags plasticMaterialFlags( const otk::pbrt::PlasticMaterial& material )
{
    MaterialFlags flags{};
    if( !material.alphaMapFileName.empty() )
        flags |= MaterialFlags::ALPHA_MAP;
    if( !material.diffuseMapFileName.empty() )
        flags |= MaterialFlags::DIFFUSE_MAP;
    return flags;
}

inline MaterialFlags shapeMaterialFlags( const otk::pbrt::ShapeDefinition& shape )
{
    return plasticMaterialFlags( shape.material );
}

}  // namespace demandPbrtScene
