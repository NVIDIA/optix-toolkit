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

#include "SphereInstances.h"

#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <OptiXToolkit/Util/Fill.h>

#include <optix_stubs.h>

namespace demandGeometryViewer {

void SphereInstances::add( float3 center, float radius, uint_t id, uint_t sbtIndex )
{
    m_centers.push_back( center );
    m_radii.push_back( radius );
    m_sphereIds.push_back( id );
    m_sbtIndices.push_back( sbtIndex );
}

void SphereInstances::remove( uint_t id )
{
    auto pos = std::find( m_sphereIds.begin(), m_sphereIds.end(), id );
    if( pos == m_sphereIds.end() )
        throw std::runtime_error( "Unknown sphere id " + std::to_string( id ) );

    const ptrdiff_t posIndex = pos - m_sphereIds.begin();
    m_centers.erase( m_centers.begin() + posIndex );
    m_radii.erase( m_radii.begin() + posIndex );
    m_sphereIds.erase( m_sphereIds.begin() + posIndex );
    m_sbtIndices.erase( m_sbtIndices.begin() + posIndex );
}

void SphereInstances::setInstanceSbtIndex( uint_t id, uint_t index )
{
    auto pos = std::find( m_sphereIds.begin(), m_sphereIds.end(), id );
    if( pos == m_sphereIds.end() )
        throw std::runtime_error( "Unknown sphere id " + std::to_string( id ) );

    const ptrdiff_t posIndex = pos - m_sphereIds.begin();
    m_sbtIndices[posIndex]   = index;
}

OptixTraversableHandle SphereInstances::createTraversable( OptixDeviceContext dc, CUstream stream )
{
    m_centers.copyToDeviceAsync( stream );
    m_radii.copyToDeviceAsync( stream );
    m_sphereIds.copyToDeviceAsync( stream );
    m_sbtIndices.copyToDeviceAsync( stream );

    const unsigned int NUM_MOTION_STEPS = 1;
    const CUdeviceptr  devVertexBuffers[NUM_MOTION_STEPS]{m_centers};
    const CUdeviceptr  devRadiiBuffers[NUM_MOTION_STEPS]{m_radii};

    const uint_t NUM_SPHERE_SBT_RECORDS =
        m_centers.empty() ? 1 : *std::max_element( m_sbtIndices.begin(), m_sbtIndices.end() ) + 1;
    std::vector<uint32_t> flags;
    flags.resize( NUM_SPHERE_SBT_RECORDS );
    otk::fill( flags, OPTIX_GEOMETRY_FLAG_NONE );

    // All the spheres are described in a single build input.
    const uint_t    NUM_BUILD_INPUTS = 1;
    OptixBuildInput sphereInput[NUM_BUILD_INPUTS]{};
    otk::BuildInputBuilder( sphereInput )
        .spheres( devVertexBuffers, static_cast<uint_t>( m_centers.size() ), devRadiiBuffers, flags.data(),
                  NUM_SPHERE_SBT_RECORDS, m_sbtIndices.devicePtr(), sizeof( uint_t ) );

    const OptixAccelBuildOptions accelOptions = {
        OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS,              // buildFlags
        OPTIX_BUILD_OPERATION_BUILD,                                                               // operation
        OptixMotionOptions{ /*numKeys=*/0, /*flags=*/0, /*timeBegin=*/0.f, /*timeEnd=*/0.f }
    };
    OptixAccelBufferSizes gasSizes{};
    OTK_ERROR_CHECK( optixAccelComputeMemoryUsage( dc, &accelOptions, sphereInput, NUM_BUILD_INPUTS, &gasSizes ) );

    m_devTempBufferGas.resize( gasSizes.tempSizeInBytes );
    m_devGeomAs.resize( gasSizes.outputSizeInBytes );
    OptixTraversableHandle traversable;
    OTK_ERROR_CHECK( optixAccelBuild( dc, stream, &accelOptions, sphereInput, NUM_BUILD_INPUTS, m_devTempBufferGas,
                                      gasSizes.tempSizeInBytes, m_devGeomAs, gasSizes.outputSizeInBytes, &traversable, nullptr, 0 ) );
    return traversable;
}

}  // namespace demandGeometryViewer
