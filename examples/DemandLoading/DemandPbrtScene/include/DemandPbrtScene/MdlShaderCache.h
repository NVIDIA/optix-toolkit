// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Config.h"

#ifdef OTK_USE_MDL

#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <cstddef>
#include <map>
#include <string>

namespace demandPbrtScene {

struct MdlShaderKey
{
    std::string signature;
};

bool operator==( const MdlShaderKey& lhs, const MdlShaderKey& rhs );
bool operator!=( const MdlShaderKey& lhs, const MdlShaderKey& rhs );
bool operator<( const MdlShaderKey& lhs, const MdlShaderKey& rhs );

std::string  toString( const MdlShaderKey& key );
MdlShaderKey makeMdlShaderKey( const otk::pbrt::PbrtMaterial& material );

struct GeneratedMdlSource
{
    std::string moduleName;
    std::string materialName;
    std::string source;
};

class MdlGeneratedSourceCache
{
  public:
    const GeneratedMdlSource& getOrCreate( const MdlShaderKey& key );
    bool                      contains( const MdlShaderKey& key ) const;
    std::size_t               size() const;
    void                      clear();

  private:
    std::map<MdlShaderKey, GeneratedMdlSource> m_sources;
};

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL
