// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Config.h"

#ifdef OTK_USE_MDL
#include "DemandPbrtScene/MdlKeyBuilder.h"
#include "DemandPbrtScene/MdlMaterialModelBuilder.h"
#include "DemandPbrtScene/MdlParameterBinder.h"
#include "DemandPbrtScene/MdlShaderCompileCacheStatistics.h"

#include <cstddef>
#include <map>
#include <string>

namespace demandPbrtScene {

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
    bool                  sourceShapeProgramReusable{};
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
    void setRecordState( const MdlMaterialInstanceKey& key, MdlShaderCompileState state, const std::string& diagnostics );

    struct SourceCompileRecord
    {
        MdlShaderCompileState state{ MdlShaderCompileState::MISSING };
        unsigned int          shaderKeyId{};
        std::string           diagnostics;
    };

    std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord> m_records;
    std::map<MdlShaderKey, unsigned int>                     m_sourceKeyUseCounts;
    std::map<MdlShaderKey, SourceCompileRecord>              m_sourceRecords;
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
