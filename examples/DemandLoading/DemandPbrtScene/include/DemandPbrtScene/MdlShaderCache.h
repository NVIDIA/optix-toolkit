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
    unsigned int          shaderKeyId{};
    std::string           diagnostics;
};

class MdlShaderCompileCache
{
  public:
    const MdlShaderCompileRecord& getRecord( const MdlShaderKey& key );
    MdlShaderCompileState         state( const MdlShaderKey& key ) const;
    unsigned int                  shaderKeyId( const MdlShaderKey& key ) const;
    std::string                   diagnostics( const MdlShaderKey& key ) const;

    bool requestCompile( const MdlShaderKey& key );
    void markCompiling( const MdlShaderKey& key );
    void markReady( const MdlShaderKey& key );
    void markFailed( const MdlShaderKey& key, const std::string& diagnostics );

    MdlShaderCompileCacheStatistics getStatistics() const;
    std::size_t                     size() const;
    void                            clear();

  private:
    MdlShaderCompileRecord& getMutableRecord( const MdlShaderKey& key );

    std::map<MdlShaderKey, MdlShaderCompileRecord> m_records;
    MdlShaderCompileCacheStatistics                m_stats{};
    unsigned int                                   m_nextShaderKeyId{ 1U };
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
