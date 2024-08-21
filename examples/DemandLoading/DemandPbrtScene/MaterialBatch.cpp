// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MaterialBatch.h"

#include "DemandPbrtScene/Params.h"

#include <OptiXToolkit/Memory/SyncVector.h>

namespace demandPbrtScene {

namespace {

class MaterialBatchImpl : public MaterialBatch
{
  public:
    ~MaterialBatchImpl() override = default;

    uint_t addPrimitiveMaterialRange( uint_t primitiveIndexEnd, uint_t materialId ) override;
    void   addMaterialIndex( uint_t numGroups, uint_t materialBegin ) override;

    void setLaunchParams( CUstream stream, Params& params ) override;

  private:
    otk::SyncVector<PrimitiveMaterialRange> m_ranges;
    otk::SyncVector<MaterialIndex>          m_indices;
};

uint_t MaterialBatchImpl::addPrimitiveMaterialRange( uint_t primitiveIndexEnd, uint_t materialId )
{
    const uint_t startIndex{ static_cast<uint_t>( m_ranges.size() ) };
    m_ranges.push_back( PrimitiveMaterialRange{ primitiveIndexEnd, materialId } );
    return startIndex;
}

void MaterialBatchImpl::setLaunchParams( CUstream stream, Params& params )
{
    m_ranges.copyToDevice();
    m_indices.copyToDevice();
    params.numPrimitiveMaterials = static_cast<uint_t>( m_ranges.size() );
    params.primitiveMaterials    = m_ranges.typedDevicePtr();
    params.numMaterialIndices    = static_cast<uint_t>( m_indices.size() );
    params.materialIndices       = m_indices.typedDevicePtr();
}

void MaterialBatchImpl::addMaterialIndex( uint_t numGroups, uint_t materialBegin )
{
    m_indices.push_back( MaterialIndex{ numGroups, materialBegin } );
}

}  // namespace

MaterialBatchPtr createMaterialBatch()
{
    return std::make_shared<MaterialBatchImpl>();
}

}  // namespace demandPbrtScene
