// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MdlShaderCache.h"

#ifdef OTK_USE_MDL

#include "DemandPbrtScene/MdlMaterialModelBuilder.h"

#include <set>
#include <utility>

namespace demandPbrtScene {

MdlShaderCompileRecord& MdlShaderCompileCache::getMutableRecord( const MdlMaterialInstanceKey& key )
{
    std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord>::iterator it = m_records.find( key );
    if( it == m_records.end() )
    {
        MdlShaderCompileRecord record{};
        record.sourceKey                                              = key.sourceKey;
        record.sourceShapeProgramReusable                             = key.sourceShapeProgramReusable;
        std::map<MdlShaderKey, unsigned int>::iterator sourceUseCount = m_sourceKeyUseCounts.find( key.sourceKey );
        if( sourceUseCount == m_sourceKeyUseCounts.end() )
        {
            m_sourceKeyUseCounts.insert( std::make_pair( key.sourceKey, 1U ) );
        }
        else
        {
            ++sourceUseCount->second;
            ++m_stats.numSourceCacheHits;
            if( key.sourceShapeProgramReusable )
            {
                ++m_stats.numShaderCacheHits;
            }
        }
        if( key.sourceShapeProgramReusable )
        {
            std::map<MdlShaderKey, SourceCompileRecord>::iterator source = m_sourceRecords.find( key.sourceKey );
            if( source == m_sourceRecords.end() )
            {
                SourceCompileRecord sourceRecord{};
                sourceRecord.shaderKeyId = m_nextShaderKeyId++;
                source = m_sourceRecords.insert( std::make_pair( key.sourceKey, sourceRecord ) ).first;
            }
            record.state       = source->second.state;
            record.shaderKeyId = source->second.shaderKeyId;
            record.diagnostics = source->second.diagnostics;
        }
        else
        {
            record.shaderKeyId = m_nextShaderKeyId++;
        }

        it = m_records.insert( std::make_pair( key, record ) ).first;
    }
    return it->second;
}

const MdlShaderCompileRecord& MdlShaderCompileCache::getRecord( const MdlMaterialInstanceKey& key )
{
    ++m_stats.numShaderRequests;
    if( m_records.find( key ) != m_records.end() )
    {
        ++m_stats.numMaterialInstanceCacheHits;
        ++m_stats.numShaderCacheHits;
    }
    return getMutableRecord( key );
}

MdlShaderCompileState MdlShaderCompileCache::state( const MdlMaterialInstanceKey& key ) const
{
    std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord>::const_iterator it = m_records.find( key );
    if( it == m_records.end() )
    {
        return MdlShaderCompileState::MISSING;
    }
    if( it->second.sourceShapeProgramReusable )
    {
        const std::map<MdlShaderKey, SourceCompileRecord>::const_iterator source = m_sourceRecords.find( it->second.sourceKey );
        if( source != m_sourceRecords.end() )
        {
            return source->second.state;
        }
    }
    return it->second.state;
}

unsigned int MdlShaderCompileCache::shaderKeyId( const MdlMaterialInstanceKey& key ) const
{
    std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord>::const_iterator it = m_records.find( key );
    return it == m_records.end() ? 0U : it->second.shaderKeyId;
}

std::string MdlShaderCompileCache::diagnostics( const MdlMaterialInstanceKey& key ) const
{
    std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord>::const_iterator it = m_records.find( key );
    if( it == m_records.end() )
    {
        return std::string{};
    }
    if( it->second.sourceShapeProgramReusable )
    {
        const std::map<MdlShaderKey, SourceCompileRecord>::const_iterator source = m_sourceRecords.find( it->second.sourceKey );
        if( source != m_sourceRecords.end() )
        {
            return source->second.diagnostics;
        }
    }
    return it->second.diagnostics;
}

bool MdlShaderCompileCache::requestCompile( const MdlMaterialInstanceKey& key )
{
    std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord>::iterator it = m_records.find( key );
    if( it != m_records.end() && it->second.state != MdlShaderCompileState::MISSING )
    {
        ++m_stats.numMaterialInstanceCacheHits;
        ++m_stats.numShaderCacheHits;
        return false;
    }

    MdlShaderCompileRecord& record = it == m_records.end() ? getMutableRecord( key ) : it->second;
    if( record.sourceShapeProgramReusable )
    {
        SourceCompileRecord& source = m_sourceRecords[record.sourceKey];
        if( source.shaderKeyId == 0U )
        {
            source.shaderKeyId = record.shaderKeyId;
        }
        if( source.state != MdlShaderCompileState::MISSING )
        {
            record.state       = source.state;
            record.diagnostics = source.diagnostics;
            return false;
        }

        source.state = MdlShaderCompileState::QUEUED;
        source.diagnostics.clear();
        record.state = MdlShaderCompileState::QUEUED;
        record.diagnostics.clear();
        ++m_stats.numCompileRequests;
        return true;
    }

    if( record.state != MdlShaderCompileState::MISSING )
    {
        return false;
    }

    record.state = MdlShaderCompileState::QUEUED;
    record.diagnostics.clear();
    ++m_stats.numCompileRequests;
    return true;
}

void MdlShaderCompileCache::setRecordState( const MdlMaterialInstanceKey& key, MdlShaderCompileState state, const std::string& diagnostics )
{
    MdlShaderCompileRecord& record = getMutableRecord( key );
    if( record.sourceShapeProgramReusable )
    {
        SourceCompileRecord& source = m_sourceRecords[record.sourceKey];
        if( source.shaderKeyId == 0U )
        {
            source.shaderKeyId = record.shaderKeyId;
        }
        if( state == MdlShaderCompileState::READY && source.state != MdlShaderCompileState::READY )
        {
            ++m_stats.numCompletedCompiles;
        }
        source.state       = state;
        source.diagnostics = diagnostics;
        for( std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord>::iterator it = m_records.begin(); it != m_records.end(); ++it )
        {
            if( it->second.sourceShapeProgramReusable && it->second.sourceKey == record.sourceKey )
            {
                it->second.state       = state;
                it->second.diagnostics = diagnostics;
            }
        }
        return;
    }

    if( state == MdlShaderCompileState::READY && record.state != MdlShaderCompileState::READY )
    {
        ++m_stats.numCompletedCompiles;
    }
    record.state       = state;
    record.diagnostics = diagnostics;
}

void MdlShaderCompileCache::markCompiling( const MdlMaterialInstanceKey& key )
{
    setRecordState( key, MdlShaderCompileState::COMPILING, std::string{} );
}

void MdlShaderCompileCache::markReady( const MdlMaterialInstanceKey& key )
{
    setRecordState( key, MdlShaderCompileState::READY, std::string{} );
}

void MdlShaderCompileCache::markFailed( const MdlMaterialInstanceKey& key, const std::string& diagnostics )
{
    setRecordState( key, MdlShaderCompileState::FAILED, diagnostics );
}

MdlShaderCompileCacheStatistics MdlShaderCompileCache::getStatistics() const
{
    MdlShaderCompileCacheStatistics stats{ m_stats };
    std::set<MdlShaderKey>          countedSourceKeys;
    for( std::map<MdlMaterialInstanceKey, MdlShaderCompileRecord>::const_iterator it = m_records.begin();
         it != m_records.end(); ++it )
    {
        MdlShaderCompileState state{ it->second.state };
        if( it->second.sourceShapeProgramReusable )
        {
            if( !countedSourceKeys.insert( it->second.sourceKey ).second )
            {
                continue;
            }
            const std::map<MdlShaderKey, SourceCompileRecord>::const_iterator source = m_sourceRecords.find( it->second.sourceKey );
            if( source != m_sourceRecords.end() )
            {
                state = source->second.state;
            }
        }
        switch( state )
        {
            case MdlShaderCompileState::MISSING:
                ++stats.numMissingShaders;
                break;
            case MdlShaderCompileState::QUEUED:
                ++stats.numQueuedShaders;
                break;
            case MdlShaderCompileState::COMPILING:
                ++stats.numCompilingShaders;
                break;
            case MdlShaderCompileState::READY:
                ++stats.numReadyShaders;
                break;
            case MdlShaderCompileState::FAILED:
                ++stats.numFailedShaders;
                break;
        }
    }
    return stats;
}

std::size_t MdlShaderCompileCache::size() const
{
    return m_records.size();
}

void MdlShaderCompileCache::clear()
{
    m_records.clear();
    m_sourceKeyUseCounts.clear();
    m_sourceRecords.clear();
    m_stats           = MdlShaderCompileCacheStatistics{};
    m_nextShaderKeyId = 1U;
}

const GeneratedMdlSource& MdlGeneratedSourceCache::getSource( const MdlShaderKey& key )
{
    std::map<MdlShaderKey, GeneratedMdlSource>::iterator it = m_sources.find( key );
    if( it == m_sources.end() )
    {
        it = m_sources.insert( std::make_pair( key, generateMdlSource( key ) ) ).first;
    }
    return it->second;
}

const GeneratedMdlSource& MdlGeneratedSourceCache::getSource( const otk::pbrt::PbrtMaterial& material )
{
    const MdlShaderKey                                   key{ makeMdlShaderKey( material ) };
    std::map<MdlShaderKey, GeneratedMdlSource>::iterator it = m_sources.find( key );
    if( it == m_sources.end() )
    {
        it = m_sources.insert( std::make_pair( key, generateMdlSource( material ) ) ).first;
    }
    return it->second;
}

bool MdlGeneratedSourceCache::contains( const MdlShaderKey& key ) const
{
    return m_sources.find( key ) != m_sources.end();
}

std::size_t MdlGeneratedSourceCache::size() const
{
    return m_sources.size();
}

void MdlGeneratedSourceCache::clear()
{
    m_sources.clear();
}

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL
