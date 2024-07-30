// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "Dependencies.h"

namespace demandPbrtScene {

struct GeometryInstance;

class ProgramGroups
{
  public:
    virtual ~ProgramGroups() = default;

    virtual void initialize() = 0;
    virtual void cleanup()    = 0;

    virtual uint_t getRealizedMaterialSbtOffset( const GeometryInstance& instance ) = 0;
};

ProgramGroupsPtr createProgramGroups( GeometryLoaderPtr geometryLoader, MaterialLoaderPtr materialLoader, RendererPtr renderer );

}  // namespace demandPbrtScene
