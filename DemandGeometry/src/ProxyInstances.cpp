//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <OptiXToolkit/DemandGeometry/ProxyInstances.h>

#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>

#include <optix_stubs.h>

#include <algorithm>
#include <cstring>

namespace {

/// Fill a std::vector<T> with a value of type U that can be converted to T.
template <typename T, typename U>
void fill( std::vector<T>& vec, U value )
{
    std::fill( std::begin( vec ), std::end( vec ), static_cast<T>( value ) );
}

}  // namespace

namespace demandGeometry {

// const uint_t                 NUM_PROXY_INSTANCES   = 1;
const uint_t                 NUM_CUSTOM_PRIMITIVES = 1;
const OptixTraversableHandle NULL_TRAVERSABLE{ 0 };

ProxyInstances::ProxyInstances( demandLoading::DemandLoader* loader )
    : m_loader( loader )
{
    // Unit cube AABB that is positioned and scaled for each proxy instance.
    const OptixAabb primitiveBounds{ 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f };
    m_primitiveBounds.push_back( primitiveBounds );
}

std::vector<uint_t> ProxyInstances::requestedProxyIds() const
{
    std::vector<uint_t> result;
    {
        std::lock_guard<std::mutex> lock( m_proxyDataMutex );
        result = m_requestedResources;
    }
    return result;
}

void ProxyInstances::clearRequestedProxyIds()
{
    std::lock_guard<std::mutex> lock( m_proxyDataMutex );

    if( m_recycleProxyIds )
    {
        m_freePages.insert( m_freePages.end(), m_requestedResources.begin(), m_requestedResources.end() );
    }
    
    m_requestedResources.clear();
}    

uint_t ProxyInstances::add( const OptixAabb& bounds )
{
    std::lock_guard<std::mutex> lock( m_proxyDataMutex );

    const uint_t index = allocateResource();
    m_proxyData.insert( m_proxyData.begin() + index, bounds );
    return m_proxyPageIds[index];
}

void ProxyInstances::remove( uint_t pageId )
{
    // Set page table entry for the requested page, ensuring that it won't be requested again.
    m_loader->setPageTableEntry( pageId, /*evictable=*/true, 0LL /* value doesn't matter */ );

    std::lock_guard<std::mutex> lock( m_proxyDataMutex );

    {
        auto pos = std::lower_bound( m_proxyPageIds.begin(), m_proxyPageIds.end(), pageId );
        if( pos == m_proxyPageIds.end() || *pos != pageId )
            throw std::runtime_error( "Resource not found for page " + std::to_string( pageId ) );

        const int index = static_cast<int>( pos - m_proxyPageIds.begin() );
        m_proxyData.erase( m_proxyData.begin() + index );
        m_proxyPageIds.erase( m_proxyPageIds.begin() + index );
    }

    {
        auto pos = std::lower_bound( m_requestedResources.begin(), m_requestedResources.end(), pageId );
        if( pos != m_requestedResources.end() )
            m_requestedResources.erase( pos );
    }

    deallocateResource( pageId );
}

void ProxyInstances::copyToDevice()
{
    m_proxyData.copyToDevice();
    m_primitiveBounds.copyToDevice();
}

void ProxyInstances::copyToDeviceAsync( CUstream stream )
{
    m_proxyData.copyToDeviceAsync( stream );
    m_primitiveBounds.copyToDeviceAsync( stream );
}

void ProxyInstances::createProxyGeomAS( OptixDeviceContext dc, CUstream stream )
{
    std::vector<uint_t> aabbInputFlags( NUM_CUSTOM_PRIMITIVES );
    fill( aabbInputFlags, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT );
    m_sbtIndices.resize( NUM_CUSTOM_PRIMITIVES );
    fill( m_sbtIndices, m_sbtIndex );
    m_sbtIndices.copyToDeviceAsync( stream );

    const uint_t                 numSbtRecords    = m_sbtIndex + 1;
    const uint_t                 NUM_BUILD_INPUTS = NUM_CUSTOM_PRIMITIVES;
    std::vector<OptixBuildInput> aabbInputs( NUM_BUILD_INPUTS );
    const unsigned int           NUM_MOTION_STEPS = 1;
    std::vector<void*>           devAabbBuffers{ NUM_MOTION_STEPS };
    devAabbBuffers[0] = m_primitiveBounds.devicePtr();
    otk::BuildInputBuilder( aabbInputs.data(), aabbInputs.size() )
        .customPrimitives( devAabbBuffers.data(), aabbInputFlags.data(), numSbtRecords, NUM_BUILD_INPUTS,
                           m_sbtIndices.devicePtr(), sizeof( uint32_t ) );

    OptixAccelBuildOptions accelOptions = {
        OPTIX_BUILD_FLAG_NONE,       // buildFlags
        OPTIX_BUILD_OPERATION_BUILD,  // operation
        OptixMotionOptions{/*numKeys=*/0, /*flags=*/0, /*timeBegin=*/0.f, /*timeEnd=*/0.f}
    };
    OptixAccelBufferSizes gasSizes{};
    OTK_ERROR_CHECK( optixAccelComputeMemoryUsage( dc, &accelOptions, aabbInputs.data(), NUM_BUILD_INPUTS, &gasSizes ) );

    m_devTempAccelBuffer.resize( gasSizes.tempSizeInBytes );
    m_devProxyGeomAccelBuffer.resize( gasSizes.outputSizeInBytes );
    OTK_ERROR_CHECK( optixAccelBuild( dc, stream, &accelOptions, aabbInputs.data(), NUM_BUILD_INPUTS, m_devTempAccelBuffer,
                                  gasSizes.tempSizeInBytes, m_devProxyGeomAccelBuffer, gasSizes.outputSizeInBytes,
                                  &m_proxyGeomTraversable, nullptr, 0 ) );
}

static void transform( float ( &result )[12], const OptixAabb& bounds )
{
    const float scaleX = bounds.maxX - bounds.minX;
    const float scaleY = bounds.maxY - bounds.minY;
    const float scaleZ = bounds.maxZ - bounds.minZ;
    // clang-format off
    const float matrix[12]{
        scaleX, 0.0f, 0.0f, bounds.minX,
        0.0f, scaleY, 0.0f, bounds.minY,
        0.0f, 0.0f, scaleZ, bounds.minZ
    };
    // clang-format on
    std::copy( std::begin( matrix ), std::end( matrix ), std::begin( result ) );
}

OptixTraversableHandle ProxyInstances::createProxyInstanceAS( OptixDeviceContext dc, CUstream stream )
{
    m_proxyInstances.clear();
    for( size_t i = 0; i < m_proxyData.size(); ++i )
    {
        OptixInstance instance{};
        transform( instance.transform, m_proxyData[i] );
        instance.instanceId        = m_proxyPageIds[i];
        instance.sbtOffset         = 0U;
        instance.visibilityMask    = 255U;
        instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = m_proxyGeomTraversable;
        m_proxyInstances.push_back( instance );
    }
    m_proxyInstances.copyToDeviceAsync( stream );

    const uint_t    NUM_BUILD_INPUTS = 1;
    OptixBuildInput inputs[NUM_BUILD_INPUTS]{};
    otk::BuildInputBuilder( inputs ).instanceArray( m_proxyInstances, static_cast<uint_t>( m_proxyData.size() ) );

    OptixAccelBuildOptions options = {
        OPTIX_BUILD_FLAG_NONE,       // buildFlags
        OPTIX_BUILD_OPERATION_BUILD,  // operation
        OptixMotionOptions{/*numKeys=*/0, /*flags=*/0, /*timeBegin=*/0.f, /*timeEnd=*/0.f}
    };
    OptixAccelBufferSizes sizes{};
    OTK_ERROR_CHECK( optixAccelComputeMemoryUsage( dc, &options, inputs, NUM_BUILD_INPUTS, &sizes ) );

    m_devTempAccelBuffer.resize( sizes.tempSizeInBytes );
    m_devProxyInstanceAccelBuffer.resize( sizes.outputSizeInBytes );
    OTK_ERROR_CHECK( optixAccelBuild( dc, stream, &options, inputs, NUM_BUILD_INPUTS, m_devTempAccelBuffer,
                                  sizes.tempSizeInBytes, m_devProxyInstanceAccelBuffer, sizes.outputSizeInBytes,
                                  &m_proxyInstanceTraversable, nullptr, 0 ) );
#ifndef NDEBG
    OTK_CUDA_SYNC_CHECK();
#endif    
    return m_proxyInstanceTraversable;
}

OptixTraversableHandle ProxyInstances::createTraversable( OptixDeviceContext dc, CUstream stream )
{
    if( m_proxyGeomTraversable == NULL_TRAVERSABLE )
    {
        createProxyGeomAS( dc, stream );
    }
    return createProxyInstanceAS( dc, stream );
}

const char* ProxyInstances::getCHFunctionName() const
{
    return "__closesthit__electricBoundingBox";
}

const char* ProxyInstances::getISFunctionName() const
{
    return "__intersection__electricBoundingBox";
}

int ProxyInstances::getNumAttributes() const
{
    const int NUM_ATTRIBUTES = 4;
    return NUM_ATTRIBUTES;
}

bool ProxyInstances::callback( CUstream /*stream*/, uint_t pageId, void** /*pageTableEntry*/ )
{
    std::lock_guard<std::mutex> lock( m_proxyDataMutex );

    {
        auto pos = std::lower_bound( m_proxyPageIds.begin(), m_proxyPageIds.end(), pageId);
        if( pos == m_proxyPageIds.end() || *pos != pageId )
            throw std::runtime_error( "Callback invoked for resource " + std::to_string( pageId )
                                      + " not associated with a proxy." );
    }

    // Deduplicate the requested resource page id.
    auto pos = std::lower_bound( m_requestedResources.begin(), m_requestedResources.end(), pageId );
    if( pos == m_requestedResources.end() || *pos != pageId )
        m_requestedResources.insert( pos, pageId );

    // The callback returns false, indicating that the request has not yet been satisfied.  Later,
    // when the proxy has been resolved, setPageTableEntry is called to update the page table.
    return false;
}

uint_t ProxyInstances::insertResource( const uint_t pageId )
{
    auto pos = std::lower_bound( m_proxyPageIds.begin(), m_proxyPageIds.end(), pageId);
    if( pos != m_proxyPageIds.end() && *pos == pageId )
        throw std::runtime_error( "Duplicate Resource found for page " + std::to_string( pageId ) );

    const uint_t index = static_cast<uint_t>( pos - m_proxyPageIds.begin() );
    m_proxyPageIds.insert( pos, pageId );
    return index;
}

uint_t ProxyInstances::allocateResource()
{
    if( m_recycleProxyIds && !m_freePages.empty() )
    {
        const uint_t pageId = m_freePages.back();
        m_freePages.pop_back();
        return insertResource( pageId );
    }

    for( PageIdRange& range : m_pageRanges )
    {
        if( range.m_used < range.m_size )
        {
            const uint_t pageId = range.m_start + range.m_used++;
            return insertResource( pageId );
        }
    }

    PageIdRange range;
    range.m_size  = PAGE_CHUNK_SIZE;
    range.m_start = m_loader->createResource( range.m_size, s_callback, this );
    range.m_used  = 1;
    m_pageRanges.push_back( range );
    return insertResource( range.m_start );
}

void ProxyInstances::deallocateResource( uint_t pageId )
{
    if( !m_recycleProxyIds )
    {
        return;
    }

    auto pos = std::lower_bound( m_freePages.begin(), m_freePages.end(), pageId );
    if( pos != m_freePages.end() && *pos == pageId )
    {
        throw std::runtime_error( "Page " + std::to_string( pageId ) + " already freed." );
    }
    m_freePages.insert( pos, pageId );
    m_loader->invalidatePage( pageId );
}

}  // namespace demandGeometry
