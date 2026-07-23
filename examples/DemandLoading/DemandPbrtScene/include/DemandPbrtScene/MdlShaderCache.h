// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Config.h"

#ifdef OTK_USE_MDL
#include "DemandPbrtScene/MdlShaderCompileCacheStatistics.h"

#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <cstddef>
#include <map>
#include <string>
#include <vector>

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

struct MdlMaterialInstanceKey
{
    MdlShaderKey sourceKey;
    std::string  signature;
};

bool operator==( const MdlMaterialInstanceKey& lhs, const MdlMaterialInstanceKey& rhs );
bool operator!=( const MdlMaterialInstanceKey& lhs, const MdlMaterialInstanceKey& rhs );
bool operator<( const MdlMaterialInstanceKey& lhs, const MdlMaterialInstanceKey& rhs );

std::string            toString( const MdlMaterialInstanceKey& key );
MdlMaterialInstanceKey makeMdlMaterialInstanceKey( const otk::pbrt::PbrtMaterial& material );
struct GeneratedMdlSource
{
    std::string              moduleName;
    std::string              materialName;
    std::string              source;
    std::vector<std::string> unsupportedReasons;
};

GeneratedMdlSource generateMdlSource( const otk::pbrt::PbrtMaterial& material );

enum class MdlShaderCompileState
{
    MISSING,
    QUEUED,
    COMPILING,
    READY,
    FAILED,
};

struct MdlShaderCompileRecord
{
    MdlShaderCompileState state{ MdlShaderCompileState::MISSING };
    MdlShaderKey          sourceKey;
    unsigned int          shaderKeyId{};
    std::string           diagnostics;
};

class MdlShaderCompileCache
{
  public:
    const MdlShaderCompileRecord& getRecord( const MdlMaterialInstanceKey& key );
    MdlShaderCompileState         state( const MdlMaterialInstanceKey& key ) const;
    unsigned int                  shaderKeyId( const MdlMaterialInstanceKey& key ) const;
    std::string                   diagnostics( const MdlMaterialInstanceKey& key ) const;

    bool requestCompile( const MdlMaterialInstanceKey& key );
    void markCompiling( const MdlMaterialInstanceKey& key );
    void markReady( const MdlMaterialInstanceKey& key );
    void markFailed( const MdlMaterialInstanceKey& key, const std::string& diagnostics );

    MdlShaderCompileCacheStatistics getStatistics() const;
    std::size_t                     size() const;
    void                            clear();

  private:
    MdlShaderCompileRecord& getMutableRecord( const MdlMaterialInstanceKey& key );

    std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord> m_records;
    std::map<MdlShaderKey, unsigned int>                     m_sourceKeyUseCounts;
    MdlShaderCompileCacheStatistics                          m_stats{};
    unsigned int                                             m_nextShaderKeyId{ 1U };
};

class MdlGeneratedSourceCache
{
  public:
    const GeneratedMdlSource& getSource( const MdlShaderKey& key );
    const GeneratedMdlSource& getSource( const otk::pbrt::PbrtMaterial& material );
    bool                      contains( const MdlShaderKey& key ) const;
    std::size_t               size() const;
    void                      clear();

  private:
    std::map<MdlShaderKey, GeneratedMdlSource> m_sources;
};

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL
