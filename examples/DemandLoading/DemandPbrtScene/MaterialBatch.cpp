// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MaterialBatch.h"

#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/SceneSyncState.h"

#include <OptiXToolkit/Memory/SyncVector.h>

namespace demandPbrtScene {

namespace {

class MaterialBatchImpl : public MaterialBatch
{
  public:
    ~MaterialBatchImpl() override = default;

    uint_t addPrimitiveMaterialRange( uint_t primitiveIndexEnd, uint_t materialId, SceneSyncState& sync ) override;
    void   addMaterialIndex( uint_t numGroups, uint_t materialBegin, SceneSyncState& sync ) override;
};

uint_t MaterialBatchImpl::addPrimitiveMaterialRange( uint_t primitiveIndexEnd, uint_t materialId, SceneSyncState& sync )
{
    const uint_t startIndex{ static_cast<uint_t>( sync.primitiveMaterials.size() ) };
    sync.primitiveMaterials.push_back( PrimitiveMaterialRange{ primitiveIndexEnd, materialId } );
    return startIndex;
}

void MaterialBatchImpl::addMaterialIndex( uint_t numGroups, uint_t materialBegin, SceneSyncState& sync )
{
    sync.materialIndices.push_back( MaterialIndex{ numGroups, materialBegin } );
}

}  // namespace

MaterialBatchPtr createMaterialBatch()
{
    return std::make_shared<MaterialBatchImpl>();
}

}  // namespace demandPbrtScene
