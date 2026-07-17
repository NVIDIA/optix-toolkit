// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Config.h"
#include "DemandPbrtScene/Dependencies.h"

namespace demandPbrtScene {

struct GeometryInstance;
struct Options;

class ProgramGroups
{
  public:
    virtual ~ProgramGroups() = default;

    virtual void initialize() = 0;
    virtual void cleanup()    = 0;

    virtual uint_t getRealizedMaterialSbtOffset( const GeometryInstance& instance ) = 0;
#ifdef OTK_USE_MDL
    virtual uint_t getFallbackMaterialSbtOffset( const GeometryInstance& instance ) = 0;
#endif
};

ProgramGroupsPtr createProgramGroups( const Options& options, GeometryLoaderPtr geometryLoader, MaterialLoaderPtr materialLoader, RendererPtr renderer );

}  // namespace demandPbrtScene
