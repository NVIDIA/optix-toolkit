// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

// gtest has to come before pbrt stuff
#include <gmock/gmock.h>

// For Bounds3f
#include <core/geometry.h>

#include <OptiXToolkit/DemandGeometry/Mocks/Matchers.h>
#include <OptiXToolkit/DemandGeometry/Mocks/OptixCompare.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Memory/BitCast.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/vec_printers.h>

#include <optix.h>

#include <cuda.h>
#include <vector_types.h>

#include <cstdint>
#include <iostream>
#include <vector>

inline OptixAabb toOptixAabb( const ::pbrt::Bounds3f& bounds )
{
    return OptixAabb{ bounds.pMin.x, bounds.pMin.y, bounds.pMin.z, bounds.pMax.x, bounds.pMax.y, bounds.pMax.z };
}

inline bool operator==( const float3& lhs, const ::pbrt::Point3f& rhs )
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

inline bool operator==( const ::pbrt::Point3f& lhs, const float3& rhs )
{
    return rhs == lhs;
}

inline bool operator!=( const float3& lhs, const ::pbrt::Point3f& rhs )
{
    return !( lhs == rhs );
}

inline bool operator!=( const ::pbrt::Point3f& lhs, const float3& rhs )
{
    return !( lhs == rhs );
}

#define OTK_TESTING_TOSTRING_ENUM_CASE( id_ )                                                                          \
    case id_:                                                                                                          \
        return #id_
inline const char* toString( OptixVertexFormat value )
{
    switch( value )
    {
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_VERTEX_FORMAT_NONE );
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_VERTEX_FORMAT_FLOAT3 );
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_VERTEX_FORMAT_FLOAT2 );
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_VERTEX_FORMAT_HALF3 );
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_VERTEX_FORMAT_HALF2 );
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_VERTEX_FORMAT_SNORM16_3 );
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_VERTEX_FORMAT_SNORM16_2 );
    }
    return "?unknown";
}
#undef OTK_TESTING_TOSTRING_ENUM_CASE

inline ::otk::testing::OptixTriangleBuildInputPredicate hasDeviceVertexCoords( const std::vector<float>& expectedCoords )
{
    return [&]( ::testing::MatchResultListener* listener, const OptixBuildInputTriangleArray& triangles ) {
        bool result{ true };
        bool first{ true };
        auto separator = [&] {
            if( !first )
                *listener << "; ";
            first = false;
            return listener;
        };
        if( triangles.numVertices != expectedCoords.size() / 3 )
        {
            *separator() << "has num vertices " << triangles.numVertices << ", expected " << expectedCoords.size() / 3;
            result = false;
        }
        if( triangles.vertexFormat != OPTIX_VERTEX_FORMAT_FLOAT3 )
        {
            *separator() << "has vertex format " << toString( triangles.vertexFormat ) << " (" << triangles.vertexFormat
                         << "), expected OPTIX_VERTEX_FORMAT_FLOAT3 (" << OPTIX_VERTEX_FORMAT_FLOAT3 << ')';
            result = false;
        }
        if( triangles.vertexBuffers[0] == CUdeviceptr{} )
        {
            *separator() << "has null vertex buffer 0";
            result = false;
        }
        if( triangles.vertexStrideInBytes != 0 )
        {
            *separator() << "has non-zero vertex stride " << triangles.vertexStrideInBytes;
            result = false;
        }
        if( !result )
            return result;

        std::vector<float3> actualVertices;
        actualVertices.resize( triangles.numVertices );
        OTK_ERROR_CHECK( cudaMemcpy( actualVertices.data(), otk::bit_cast<void*>( triangles.vertexBuffers[0] ),
                                     sizeof( float3 ) * triangles.numVertices, cudaMemcpyDeviceToHost ) );
        for( int i = 0; i < static_cast<int>( triangles.numVertices ); ++i )
        {
            const float3 expectedVertex =
                make_float3( expectedCoords[i * 3 + 0], expectedCoords[i * 3 + 1], expectedCoords[i * 3 + 2] );
            if( actualVertices[i] != expectedVertex )
            {
                *separator() << "has vertex[" << i << "] " << actualVertices[i] << ", expected " << expectedVertex;
                result = false;
            }
        }

        return result;
    };
}

template <typename T>
inline ::otk::testing::OptixTriangleBuildInputPredicate hasDeviceIndices( const std::vector<T>& expectedIndices )
{
    return [&]( ::testing::MatchResultListener* result_listener, const OptixBuildInputTriangleArray& triangles ) {
        bool result    = true;
        auto separator = [&] {
            if( !result )
                *result_listener << "; ";
            return result_listener;
        };
        if( triangles.numIndexTriplets * 3 != expectedIndices.size() )
        {
            *separator() << " has index triplet count " << triangles.numIndexTriplets << ", expected "
                         << expectedIndices.size() / 3;
            result = false;
        }
        if( triangles.indexFormat != OPTIX_INDICES_FORMAT_UNSIGNED_INT3 )
        {
            *separator() << " has index format " << triangles.indexFormat << ", expected OPTIX_INDICES_FORMAT_UNSIGNED_INT3 ("
                         << OPTIX_INDICES_FORMAT_UNSIGNED_INT3 << ')';
            result = false;
        }
        if( triangles.indexBuffer == CUdeviceptr{} )
        {
            *separator() << " has null index buffer";
            result = false;
        }
        if( triangles.indexStrideInBytes != 0 )
        {
            *separator() << " has non-zero index stride " << triangles.indexStrideInBytes;
            result = false;
        }
        if( !result )
            return result;

        std::vector<uint3> actualIndices;
        actualIndices.resize( expectedIndices.size() );
        OTK_ERROR_CHECK( cudaMemcpy( actualIndices.data(), otk::bit_cast<void*>( triangles.indexBuffer ),
                                     sizeof( uint3 ) * triangles.numIndexTriplets, cudaMemcpyDeviceToHost ) );
        for( unsigned int i = 0; i < triangles.numIndexTriplets; ++i )
        {
            if( actualIndices[i].x != expectedIndices[i * 3 + 0] )
            {
                *separator() << " triangle[" << i << "] has index[0] " << actualIndices[i].x << ", expected "
                             << expectedIndices[i * 3 + 0];
                result = false;
            }
            if( actualIndices[i].y != ( expectedIndices )[i * 3 + 1] )
            {
                *separator() << " triangle[" << i << "] has index[1] " << actualIndices[i].y << ", expected "
                             << expectedIndices[i * 3 + 1];
                result = false;
            }
            if( actualIndices[i].z != ( expectedIndices )[i * 3 + 2] )
            {
                *separator() << " triangle[" << i << "] has index[2] " << actualIndices[i].z << ", expected "
                             << expectedIndices[i * 3 + 2];
                result = false;
            }
        }

        return result;
    };
}
