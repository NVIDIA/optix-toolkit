// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Config.h"

#ifdef OTK_USE_MDL

#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <memory>
#include <string>

namespace demandPbrtScene {

struct GeneratedMdlSource;

struct MdlTextureLookup
{
    std::string                   graphKey;
    const otk::pbrt::PbrtTexture* texture{};
};

MdlTextureLookup findMdlTexture( const otk::pbrt::PbrtMaterialGraph& graph,
                                 const std::string&                  textureName,
                                 const std::string&                  preferredValueType );

class MdlTextureGraphGenerator
{
  public:
    MdlTextureGraphGenerator( const otk::pbrt::PbrtMaterialGraph& graph, GeneratedMdlSource& result );
    ~MdlTextureGraphGenerator();

    MdlTextureGraphGenerator( const MdlTextureGraphGenerator& )            = delete;
    MdlTextureGraphGenerator& operator=( const MdlTextureGraphGenerator& ) = delete;

    std::string materialColorExpression( const ::pbrt::ParamSet& params,
                                         const std::string&      paramName,
                                         const std::string&      preferredValueType,
                                         const std::string&      defaultExpression );
    std::string materialFloatExpression( const ::pbrt::ParamSet& params,
                                         const std::string&      paramName,
                                         const std::string&      preferredValueType,
                                         const std::string&      defaultExpression );
    std::string sourcePreamble() const;
    std::string functionDefinitions() const;

  private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL

