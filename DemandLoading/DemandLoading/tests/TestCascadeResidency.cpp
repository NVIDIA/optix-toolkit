// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Tests to diagnose cascading texture residency issues.
// When REQUEST_CASCADE is defined, requestCascade() on the device can prevent
// textures from ever becoming resident.  These tests exercise the host-side
// cascade logic that interacts with that code path.

#include "PageTableManager.h"

#include <OptiXToolkit/DemandLoading/TextureCascade.h>
#include <OptiXToolkit/DemandLoading/TextureSampler.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

using namespace demandLoading;

// ---------------------------------------------------------------------------
// TestCascadePageAddressing
// Verify that the device-side cascade page calculation
//   cascadeStartIndex = pageTable.capacity
//   cascadePage = cascadeStartIndex + textureId * NUM_CASCADES + cascadeLevel
// matches the host-side reservation via PageTableManager::reserveUnbackedPages.
// ---------------------------------------------------------------------------

class DummyCascadeHandler : public RequestHandler
{
  public:
    void fillRequest( CUstream /*stream*/, unsigned int /*pageId*/ ) override {}
};

class TestCascadePageAddressing : public testing::Test
{
  protected:
    // Mirror the default option values
    static constexpr unsigned int numPages            = 64u * 1024u * 1024u;
    static constexpr unsigned int numPageTableEntries  = 1024u * 1024u;
    static constexpr unsigned int maxTextures          = 256u * 1024u;
};

TEST_F( TestCascadePageAddressing, cascadeStartMatchesPageTableCapacity )
{
    // Host: allocate sampler pages (backed), then cascade pages (unbacked)
    PageTableManager    mgr( numPages, numPageTableEntries );
    DummyCascadeHandler samplerHandler;
    DummyCascadeHandler cascadeHandler;

    mgr.reserveBackedPages( maxTextures * NUM_PAGES_PER_TEXTURE, &samplerHandler );
    unsigned int cascadeStartPage = mgr.reserveUnbackedPages( NUM_CASCADES * maxTextures, &cascadeHandler );

    // Device uses pageTable.capacity (== numPageTableEntries) as cascadeStartIndex
    unsigned int deviceCascadeStart = numPageTableEntries;

    EXPECT_EQ( deviceCascadeStart, cascadeStartPage )
        << "Device cascadeStartIndex must equal host cascadeStartPage";
}

TEST_F( TestCascadePageAddressing, cascadePageForTexture )
{
    PageTableManager    mgr( numPages, numPageTableEntries );
    DummyCascadeHandler handler;

    mgr.reserveBackedPages( maxTextures * NUM_PAGES_PER_TEXTURE, &handler );
    unsigned int cascadeStartPage = mgr.reserveUnbackedPages( NUM_CASCADES * maxTextures, &handler );

    // Verify that host handler lookup matches the device page calculation
    const unsigned int textureId    = 42;
    const unsigned int cascadeLevel = 3;
    unsigned int       devicePage   = cascadeStartPage + textureId * NUM_CASCADES + cascadeLevel;

    EXPECT_EQ( &handler, mgr.getRequestHandler( devicePage ) );
}

// ---------------------------------------------------------------------------
// TestCascadeRequestFilterLogic
// The CascadeRequestFilter keeps the largest cascade per texture and knocks
// out sampler/tile requests for textures that have a pending cascade.
// These tests exercise that logic using raw page id arithmetic, independent
// of DemandLoaderImpl.
// ---------------------------------------------------------------------------

class TestCascadeRequestFilterLogic : public testing::Test
{
  protected:
    static constexpr unsigned int cascadeStart = 1024u * 1024u;  // == numPageTableEntries
    static constexpr unsigned int maxTextures  = 256u * 1024u;

    // Helper: compute cascade page id (mirrors device-side calculation)
    static unsigned int cascadePage( unsigned int textureId, unsigned int level )
    {
        return cascadeStart + textureId * NUM_CASCADES + level;
    }

    // Helper: extract texture id from cascade page
    static unsigned int pageToTextureId( unsigned int pageId )
    {
        return ( pageId - cascadeStart ) / NUM_CASCADES;
    }

    // Helper: extract cascade level from cascade page
    static unsigned int pageToCascadeLevel( unsigned int pageId )
    {
        return ( pageId - cascadeStart ) % NUM_CASCADES;
    }
};

TEST_F( TestCascadeRequestFilterLogic, roundTripTextureIdAndLevel )
{
    for( unsigned int texId = 0; texId < 10; ++texId )
    {
        for( unsigned int level = 0; level < NUM_CASCADES; ++level )
        {
            unsigned int page = cascadePage( texId, level );
            EXPECT_EQ( texId, pageToTextureId( page ) );
            EXPECT_EQ( level, pageToCascadeLevel( page ) );
        }
    }
}

TEST_F( TestCascadeRequestFilterLogic, filterKeepsLargestCascadePerTexture )
{
    // Simulate the filter's dedup: sort cascade pages, iterate in reverse,
    // keep only the largest (highest page id) per texture.
    std::vector<unsigned int> cascadePages;
    // Texture 5 requests cascade levels 1, 3, 5
    cascadePages.push_back( cascadePage( 5, 1 ) );
    cascadePages.push_back( cascadePage( 5, 3 ) );
    cascadePages.push_back( cascadePage( 5, 5 ) );
    // Texture 10 requests cascade levels 2, 4
    cascadePages.push_back( cascadePage( 10, 2 ) );
    cascadePages.push_back( cascadePage( 10, 4 ) );
    std::sort( cascadePages.begin(), cascadePages.end() );

    // Correct filter: keep highest level per texture
    std::vector<unsigned int> filtered;
    for( int i = static_cast<int>( cascadePages.size() ) - 1; i >= 0; --i )
    {
        if( filtered.empty() || pageToTextureId( cascadePages[i] ) != pageToTextureId( filtered.back() ) )
            filtered.push_back( cascadePages[i] );
    }

    // Should keep exactly 2 entries: one per texture
    // Iterating in reverse order: tex 10 has higher page ids, so it's first
    ASSERT_EQ( 2u, filtered.size() );
    EXPECT_EQ( 10u, pageToTextureId( filtered[0] ) );
    EXPECT_EQ( 4u, pageToCascadeLevel( filtered[0] ) );   // tex 10's highest
    EXPECT_EQ( 5u, pageToTextureId( filtered[1] ) );
    EXPECT_EQ( 5u, pageToCascadeLevel( filtered[1] ) );   // tex 5's highest

    // BUG DEMONSTRATION: the actual filter compares textureId with pageId
    // This simulates the buggy comparison in CascadeRequestFilter::filter()
    std::vector<unsigned int> buggyFiltered;
    for( int i = static_cast<int>( cascadePages.size() ) - 1; i >= 0; --i )
    {
        // Bug: comparing textureId (small) with page id (large) — always unequal
        if( buggyFiltered.empty() || pageToTextureId( cascadePages[i] ) != buggyFiltered.back() )
            buggyFiltered.push_back( cascadePages[i] );
    }

    // Buggy filter keeps ALL cascade requests instead of deduplicating
    EXPECT_GT( buggyFiltered.size(), filtered.size() )
        << "Buggy filter should keep more entries than correct filter (no dedup)";
    EXPECT_EQ( cascadePages.size(), buggyFiltered.size() )
        << "Buggy filter keeps all cascade pages — no deduplication occurs";
}

// ---------------------------------------------------------------------------
// TestRequestCascadeDecision
// Exercise the decision logic of requestCascade() (device code) as pure
// host-side arithmetic.  requestCascade returns true when a cascade is
// requested (forcing *isResident = false), false otherwise.
// ---------------------------------------------------------------------------

class TestRequestCascadeDecision : public testing::Test
{
  protected:
    // Simplified host-side replica of the requestCascade decision.
    // Returns true when a cascade would be requested (isResident forced false).
    static bool wouldRequestCascade( const TextureSampler* sampler, float mipLevel )
    {
        if( sampler && !sampler->hasCascade )
            return false;
        if( sampler && mipLevel >= 0.0f )
            return false;
        // Null sampler with mipLevel >= 0 falls through — still requests cascade
        return true;
    }
};

TEST_F( TestRequestCascadeDecision, samplerWithNoCascadeReturnsFalse )
{
    TextureSampler sampler{};
    sampler.hasCascade = 0;
    EXPECT_FALSE( wouldRequestCascade( &sampler, -1.0f ) );
    EXPECT_FALSE( wouldRequestCascade( &sampler, 0.0f ) );
    EXPECT_FALSE( wouldRequestCascade( &sampler, 1.0f ) );
}

TEST_F( TestRequestCascadeDecision, samplerWithCascadeAndAdequateSizeReturnsFalse )
{
    TextureSampler sampler{};
    sampler.hasCascade = 1;
    // mipLevel >= 0 means current cascade is big enough
    EXPECT_FALSE( wouldRequestCascade( &sampler, 0.0f ) );
    EXPECT_FALSE( wouldRequestCascade( &sampler, 2.0f ) );
}

TEST_F( TestRequestCascadeDecision, samplerWithCascadeAndInadequateSizeReturnsTrue )
{
    TextureSampler sampler{};
    sampler.hasCascade = 1;
    // mipLevel < 0 means we need a bigger cascade
    EXPECT_TRUE( wouldRequestCascade( &sampler, -0.5f ) );
    EXPECT_TRUE( wouldRequestCascade( &sampler, -3.0f ) );
}

TEST_F( TestRequestCascadeDecision, nullSamplerAlwaysReturnsTrue )
{
    // When sampler is null, requestCascade always returns true — even when
    // mipLevel >= 0 (which should mean no cascade is needed).
    // This means *isResident is forced false every frame the sampler is null.
    EXPECT_TRUE( wouldRequestCascade( nullptr, 0.0f ) );
    EXPECT_TRUE( wouldRequestCascade( nullptr, 5.0f ) );
    EXPECT_TRUE( wouldRequestCascade( nullptr, -1.0f ) );
}

// ---------------------------------------------------------------------------
// TestCascadeKnockoutStarvation
// Demonstrates that when a cascade request exists, the knockout logic removes
// the sampler page request, which can prevent the sampler from ever loading.
// ---------------------------------------------------------------------------

TEST( TestCascadeKnockoutStarvation, samplerPageKnockedOutByCascadeRequest )
{
    // Simulated page ids
    const unsigned int numPageTableEntries = 1024u * 1024u;
    const unsigned int maxTextures         = 256u * 1024u;
    const unsigned int textureId           = 7;
    const unsigned int samplerPage         = textureId;  // sampler page = textureId
    const unsigned int baseColorPage       = samplerIdToBaseColorId( textureId, maxTextures );
    const unsigned int cascadePageId       = numPageTableEntries + textureId * NUM_CASCADES + 0;

    // Simulate requests from device: sampler + cascade (as happens when sampler is NULL)
    std::vector<unsigned int> deviceRequests = { samplerPage, baseColorPage, cascadePageId };

    // The knockout map marks ranges to suppress
    std::map<unsigned int, unsigned int> knockoutPages;
    knockoutPages[0] = 0;  // sentinel

    // Cascade request causes knockout of sampler and base color pages
    knockoutPages[textureId]   = textureId + 1;
    knockoutPages[baseColorPage] = baseColorPage + 1;
    // (In practice, tile pages would also be knocked out)

    // Apply knockout to non-cascade requests
    std::vector<unsigned int> survivingRequests;
    for( unsigned int req : deviceRequests )
    {
        // Skip cascade pages (handled separately)
        if( req >= numPageTableEntries )
            continue;

        auto it = knockoutPages.upper_bound( req );
        if( it != knockoutPages.begin() )
            --it;
        if( req >= it->first && req < it->second )
            continue;  // knocked out

        survivingRequests.push_back( req );
    }

    // Both sampler and base color pages are knocked out
    EXPECT_TRUE( survivingRequests.empty() )
        << "Sampler and base color requests are knocked out by cascade request";

    // This means: every frame the sampler is NULL, requestCascade fires (null
    // sampler always returns true), the cascade filter knocks out the sampler
    // request, and the sampler never loads.  The texture stays non-resident.
}

// ---------------------------------------------------------------------------
// TestCascadeLevelToTextureSize
// Verify CascadeRequestHandler helper arithmetic.
// ---------------------------------------------------------------------------

TEST( TestCascadeLevelToTextureSize, cascadeSizes )
{
    // CASCADE_BASE << level for levels 0..NUM_CASCADES-2
    EXPECT_EQ( CASCADE_BASE, CASCADE_BASE << 0 );
    EXPECT_EQ( 128u, CASCADE_BASE << 1 );
    EXPECT_EQ( 256u, CASCADE_BASE << 2 );
    EXPECT_EQ( 512u, CASCADE_BASE << 3 );

    // The last cascade level maps to MAX_TEXTURE_SIZE (65536)
    const unsigned int MAX_TEXTURE_SIZE = 65536;
    unsigned int lastLevel = NUM_CASCADES - 1;
    unsigned int size = ( lastLevel >= NUM_CASCADES - 1 ) ? MAX_TEXTURE_SIZE : CASCADE_BASE << lastLevel;
    EXPECT_EQ( MAX_TEXTURE_SIZE, size );
}
