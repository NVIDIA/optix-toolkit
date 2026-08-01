// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Config.h"
#include "DemandPbrtScene/Dependencies.h"
#ifdef OTK_USE_MDL
#include "DemandPbrtScene/Params.h"

#include <stdexcept>
#include <string>
#endif

namespace demandPbrtScene {

struct GeometryInstance;
#ifdef OTK_USE_MDL
struct FourierBsdfTable;
#endif
struct Options;

#ifdef OTK_USE_MDL
class MdlMaterialBuildPending : public std::runtime_error
{
  public:
    explicit MdlMaterialBuildPending( const std::string& message )
        : std::runtime_error( message )
    {
    }
};
#endif

class ProgramGroups
{
  public:
    virtual ~ProgramGroups() = default;

    virtual void initialize() = 0;
    virtual void cleanup()    = 0;

    virtual uint_t getRealizedMaterialSbtOffset( const GeometryInstance& instance ) = 0;
#ifdef OTK_USE_MDL
    virtual uint_t            getFallbackMaterialSbtOffset( const GeometryInstance& instance )                 = 0;
    virtual uint_t            getMdlMaterialSbtOffset( const GeometryInstance& instance )                      = 0;
    virtual uint_t            getFourierMaterialSbtOffset( const GeometryInstance& instance )                  = 0;
    virtual MdlMaterialShader realizeMdlMaterialShader( const GeometryInstance& instance, uint_t shaderKeyId ) = 0;
    virtual FourierMaterialResource realizeFourierMaterialResource( const GeometryInstance& instance,
                                                                    const FourierBsdfTable& table )            = 0;
#endif
};

ProgramGroupsPtr createProgramGroups( const Options& options, GeometryLoaderPtr geometryLoader, MaterialLoaderPtr materialLoader, RendererPtr renderer );

}  // namespace demandPbrtScene
