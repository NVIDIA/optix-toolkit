// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Dependencies.h"
#include "DemandPbrtScene/Params.h"

#include <OptiXToolkit/Memory/SyncVector.h>

#include <optix.h>

namespace demandPbrtScene {

/// Scene state that needs to be synchronized to the launch Params before launch
struct SceneSyncState
{
    uint_t                            minAlphaTextureId{ ~0U };
    uint_t                            maxAlphaTextureId{};
    uint_t                            minDiffuseTextureId{ ~0U };
    uint_t                            maxDiffuseTextureId{};
    otk::SyncVector<OptixInstance>    topLevelInstances;    // OptixInstance array for building TLIAS
    otk::SyncVector<PartialMaterial>  partialMaterials;     // indexed by materialId
    otk::SyncVector<TriangleUVs*>     partialUVs;           // indexed by materialId
    otk::SyncVector<uint_t>           instanceMaterialIds;  // indexed by instance id
    otk::SyncVector<PhongMaterial>    realizedMaterials;    // indexed by values in instanceMaterialIds
    otk::SyncVector<TriangleNormals*> realizedNormals;      // indexed by instance id, then by primitive index
    otk::SyncVector<TriangleUVs*>     realizedUVs;          // indexed by instance id, then by primitive index
    otk::SyncVector<DirectionalLight> directionalLights;    // defined by the scene
    otk::SyncVector<InfiniteLight>    infiniteLights;       // defined by the scene
    otk::SyncVector<MaterialIndex>    materialIndices;      // indexed by instanceId, one entry per instance
    otk::SyncVector<PrimitiveMaterialRange> primitiveMaterials;  // indexed by MaterialIndex::primitiveMaterialBegin; one entry per material group per instance
};

}  // namespace demandPbrtScene
