// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>

#include <memory>

namespace demandPbrtScene {

struct Params;

using uint_t = unsigned int;

/// MaterialBatch
///
/// To support multiple materials for a single GAS, we must know the primitive indices
/// that correspond to a material id.  This information is stored in the Params via
/// an array of MaterialIndex and an array of PrimitiveMaterialRange structures.
///
/// There is one MaterialIndex per instance, indexed by the instance index.  This allows
/// the shader to know how many material groups are associated with the GAS in the instance.
/// A range of entries in the PrimitiveMaterialRange array correspond to the material groups
/// in the instance, one per group.  Each PrimitiveMaterialRange entry indicates the ending
/// primitive index for the material and the associated material id.
///
/// addPrimitiveMaterialRange is used to build an entry in the PrimitiveMaterialRange array
/// and returns the associated array index.
///
/// addMaterialIndex is used to add the MaterialIndex information for the instance.
///
/// setLaunchParams is used to update the launch parameters with appropriate array pointers
/// and counts.
///
class MaterialBatch
{
  public:
    virtual ~MaterialBatch() = default;

    virtual uint_t addPrimitiveMaterialRange( uint_t primitiveIndexEnd, uint_t materialId ) = 0;
    virtual void   addMaterialIndex( uint_t numGroups, uint_t materialBegin )               = 0;
    virtual void   setLaunchParams( CUstream stream, Params& params )                       = 0;
};

using MaterialBatchPtr = std::shared_ptr<MaterialBatch>;

MaterialBatchPtr createMaterialBatch();

}  // namespace demandPbrtScene
