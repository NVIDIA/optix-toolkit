// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>

#include <memory>

namespace demandPbrtScene {

struct Params;

using uint_t = unsigned int;

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
