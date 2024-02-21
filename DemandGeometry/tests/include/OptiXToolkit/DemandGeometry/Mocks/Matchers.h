//
//  Copyright (c) 2023 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#pragma once

#include <OptiXToolkit/DemandGeometry/Mocks/OptixCompare.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Memory/BitCast.h>

#include <optix.h>

#include <gmock/gmock.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <string>

// These matchers allow you to express expectations on OptiX structures passed as arguments
// to API calls.
//
// Because some structures, such as an OptixBuildInput, are discriminated unions, expressing
// the expectations on values deep within the structures can be complex.
//
// The functions and matchers in this header are used to express such complex relationships
// in a hopefully more fluid manner.
//
// There are two kinds of entities created here: ListenerPredicates and matchers.
// The listener predicates are strictly typed against their corresponding structure
// and are used to validate the contents of the structure.  The predicates are combined
// with hasAll() to appear as a single predicate.
//
// The build input matchers take a predicate that is applied against the appropriate union
// member for the type of the build input.
//
// Because some build inputs are copied to the device, the hasDevice... predicates first
// copy the input from the device into host memory and then apply further predicates to
// that host memory.  With hasAll(), an arbitrary number of predicates can be applied to
// the data.

namespace otk {
namespace testing {

// A ListenerPredicate performs some test against an instance of T and reports
// messages to the MatchResultListener on the result of the test.
template <typename T>
using ListenerPredicate = std::function<bool( ::testing::MatchResultListener*, const T& )>;

using OptixBuildInputPredicate                = ListenerPredicate<OptixBuildInput>;
using OptixCustomPrimitiveBuildInputPredicate = ListenerPredicate<OptixBuildInputCustomPrimitiveArray>;
using OptixInstanceBuildInputPredicate        = ListenerPredicate<OptixBuildInputInstanceArray>;
using OptixInstancePredicate                  = ListenerPredicate<OptixInstance>;
using OptixInstanceVectorPredicate            = ListenerPredicate<std::vector<OptixInstance>>;
using OptixSphereBuildInputPredicate          = ListenerPredicate<OptixBuildInputSphereArray>;
using OptixTriangleBuildInputPredicate        = ListenerPredicate<OptixBuildInputTriangleArray>;

// hasAll constructs a lambda that evaluates all the predicates; note that shortcut
// evaluation is inhibited because we want all the messages from all the failed
// predicates to be reported.
//
// If we used shortcut evaluation, then the first failing predicate would prevent
// the other predicates from being evaluated.  This would imply re-running the
// tests to fix the errors one by one.
//
template <typename T>
ListenerPredicate<T> hasAll( const ListenerPredicate<T>& head )
{
    return head;
}

template <typename T, typename... Predicates>
ListenerPredicate<T> hasAll( const ListenerPredicate<T>& head, Predicates... tail )
{
    return [=]( ::testing::MatchResultListener* listener, const T& instance ) {
        const bool headResult = head( listener, instance );
        *listener << "; ";
        const bool tailResult = hasAll( tail... )( listener, instance );
        return headResult && tailResult;
    };
}

// A listener predicate that matches any argument.
template <typename T>
ListenerPredicate<T> any()
{
    return [=]( ::testing::MatchResultListener* listener, const T& ) {
        *listener << "any";
        return true;
    };
}

// A listener predicate that doesn't match any argument.
template <typename T>
ListenerPredicate<T> none()
{
    return [=]( ::testing::MatchResultListener* listener, const T& ) {
        *listener << "none";
        return false;
    };
}

#define OTK_TESTING_TOSTRING_ENUM_CASE( id_ )                                                                          \
    case id_:                                                                                                          \
        return #id_
inline const char* toString( OptixBuildInputType type )
{
    switch( type )
    {
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_BUILD_INPUT_TYPE_TRIANGLES );
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES );
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_BUILD_INPUT_TYPE_INSTANCES );
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS );
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_BUILD_INPUT_TYPE_CURVES );
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_BUILD_INPUT_TYPE_SPHERES );
    }
    return "?unknown OptixBuildInputType";
}
inline const char* toString( OptixTransformFormat value )
{
    switch( value )
    {
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_TRANSFORM_FORMAT_NONE );
        OTK_TESTING_TOSTRING_ENUM_CASE( OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 );
    }
    return "?unknown OptixTransformFormat";
}
#undef OTK_TESTING_TOSTRING_ENUM_CASE

template <typename T>
bool hasEqualValues( ::testing::MatchResultListener* listener, const char* label, T expected, T actual )
{
    if( expected != actual )
    {
        *listener << "has " << label << actual << ", expected " << expected;
        return false;
    }

    *listener << "has " << label << ' ' << expected;
    return true;
}

inline OptixInstanceBuildInputPredicate hasNumInstances( unsigned int num )
{
    return [=]( ::testing::MatchResultListener* listener, const OptixBuildInputInstanceArray& instances ) {
        return hasEqualValues( listener, "num instances", num, instances.numInstances );
    };
}

template <typename InputIter>
std::string getRangeDifferences( InputIter begin, InputIter end, const float* rhs, const std::function<bool( float, float )>& compare )
{
    std::string result;
    int         index{};
    while( begin != end )
    {
        if( !compare( *begin, *rhs ) )
        {
            if( !result.empty() )
                result += ", ";
            result += "index " + std::to_string( index ) + ' ' + std::to_string( *begin ) + " != " + std::to_string( *rhs );
        }
        ++begin;
        ++rhs;
        ++index;
    }
    return result;
}

inline std::string toString( const std::array<float, 12>& transform )
{
    std::string result{ "{ " };
    bool        firstRow{ true };
    for( int row = 0; row < 3; ++row )
    {
        if( !firstRow )
        {
            result += ", ";
        }
        result += "{ ";
        bool firstCol{ true };
        for( int col = 0; col < 4; ++col )
        {
            if( !firstCol )
            {
                result += ", ";
            }
            result += std::to_string( transform[row * 4 + col] );
            firstCol = false;
        }
        result += " }";
        firstRow = false;
    }
    result += " }";
    return result;
}

inline OptixInstancePredicate hasInstanceTransform( const std::array<float, 12>& expectedTransform )
{
    return [expectedTransform]( ::testing::MatchResultListener* listener, const OptixInstance& instance ) {
        auto compare = []( float lhs, float rhs ) { return std::abs( rhs - lhs ) < 1.0e-6f; };
        if( !std::equal( expectedTransform.begin(), expectedTransform.end(), std::begin( instance.transform ), compare ) )
        {
            *listener << "has mismatched transform "
                      << getRangeDifferences( expectedTransform.begin(), expectedTransform.end(),
                                              std::begin( instance.transform ), compare );
            return false;
        }
        *listener << "has transform " << toString( expectedTransform );
        return true;
    };
}

inline OptixInstancePredicate hasInstanceId( unsigned int id )
{
    return [=]( ::testing::MatchResultListener* listener, const OptixInstance& instance ) {
        return hasEqualValues( listener, "instance id", id, instance.instanceId );
    };
}

inline OptixInstancePredicate hasInstanceTraversable( OptixTraversableHandle traversable )
{
    return [traversable]( ::testing::MatchResultListener* listener, const OptixInstance& instance ) {
        return hasEqualValues( listener, "traversable", traversable, instance.traversableHandle );
    };
}

inline OptixInstancePredicate hasInstanceSbtOffset( unsigned int offset )
{
    return [offset]( ::testing::MatchResultListener* listener, const OptixInstance& instance ) {
        return hasEqualValues( listener, "SBT offset", offset, instance.sbtOffset );
    };
}

// Apply predicates to a specific OptixInstance from a vector of instances.
template <typename... Predicates>
OptixInstanceVectorPredicate hasInstance( unsigned int index, const Predicates&... preds )
{
    return [=]( ::testing::MatchResultListener* listener, const std::vector<OptixInstance>& instances ) {
        if( index > instances.size() )
        {
            *listener << "instance index " << index << " exceeds " << instances.size();
            return false;
        }

        return hasAll( preds... )( listener, instances[index] );
    };
}

// Apply predicates to the array of OptixInstance structures copied to the device.
template <typename... Predicates>
OptixInstanceBuildInputPredicate hasDeviceInstances( const Predicates&... preds )
{
    return [=]( ::testing::MatchResultListener* listener, const OptixBuildInputInstanceArray& instances ) {
        std::vector<OptixInstance> actualInstances;
        actualInstances.resize( instances.numInstances );
        OTK_ERROR_CHECK( cudaMemcpy( actualInstances.data(), otk::bit_cast<void*>( instances.instances ),
                                     sizeof( OptixInstance ) * instances.numInstances, cudaMemcpyDeviceToHost ) );
        return hasAll( preds... )( listener, actualInstances );
    };
}

MATCHER_P2( hasInstanceBuildInput, n, predicate, "" )
{
    if( arg[n].type != OPTIX_BUILD_INPUT_TYPE_INSTANCES )
    {
        *result_listener << "input " << n << " is of type " << toString( arg[n].type ) << " (" << arg[n].type
                         << "), expected OPTIX_BUILD_INPUT_TYPE_INSTANCES (" << OPTIX_BUILD_INPUT_TYPE_INSTANCES << ')';
        return false;
    }

    *result_listener << "input " << n << " is of type OPTIX_BUILD_INPUT_TYPE_INSTANCES ("
                     << OPTIX_BUILD_INPUT_TYPE_INSTANCES << ')';
    *result_listener << "; ";
    return predicate( result_listener, arg[n].instanceArray );
}

inline OptixTriangleBuildInputPredicate hasSbtFlags( const std::vector<unsigned int>& expectedFlags )
{
    return [&]( ::testing::MatchResultListener* listener, const OptixBuildInputTriangleArray& triangles ) {
        bool result{ true };
        auto separator{ [&] {
            if( !result )
                *listener << "; ";
            return listener;
        } };
        if( triangles.numSbtRecords != expectedFlags.size() )
        {
            *separator() << "has num SBT records " << triangles.numSbtRecords << ", expected " << expectedFlags.size();
            result = false;
        }
        for( unsigned int i = 0; i < triangles.numSbtRecords; ++i )
        {
            if( triangles.flags[i] != expectedFlags[i] )
            {
                *separator() << "SBT record has flags[" << i << "] " << triangles.flags[i] << ", expected " << expectedFlags[i];
                result = false;
            }
        }
        if( result )
        {
            *listener << "has SBT flags [ ";
            bool first{ true };
            for( unsigned int flag : expectedFlags )
            {
                if( !first )
                    *listener << ", ";
                *listener << flag;
                first = false;
            }
            *listener << " ]";
        }
        return result;
    };
}

inline OptixTriangleBuildInputPredicate hasNoPreTransform()
{
    return []( ::testing::MatchResultListener* listener, const OptixBuildInputTriangleArray& triangles ) {
        bool result    = true;
        auto separator = [&] {
            if( !result )
                *listener << "; ";
            return listener;
        };
        if( triangles.transformFormat != OPTIX_TRANSFORM_FORMAT_NONE )
        {
            *separator() << "has transform format " << toString( triangles.transformFormat ) << " (" << triangles.transformFormat
                         << "), expected OPTIX_TRANSFORM_FORMAT_NONE (" << OPTIX_TRANSFORM_FORMAT_NONE << ')';
            result = false;
        }
        if( triangles.preTransform != CUdeviceptr{} )
        {
            *separator() << "has non-null preTransform " << triangles.preTransform;
            result = false;
        }
        if( result )
        {
            *listener << "has no preTransform";
        }
        return result;
    };
}

inline OptixTriangleBuildInputPredicate hasNoSbtIndexOffsets()
{
    return []( ::testing::MatchResultListener* listener, const OptixBuildInputTriangleArray& triangles ) {
        bool result    = true;
        auto separator = [&] {
            if( !result )
                *listener << "; ";
            return listener;
        };
        if( triangles.sbtIndexOffsetBuffer != CUdeviceptr{} )
        {
            *separator() << "has non-null sbt index offset buffer " << triangles.sbtIndexOffsetBuffer;
            result = false;
        }
        if( triangles.sbtIndexOffsetSizeInBytes != 0 )
        {
            *separator() << "has non-zero sbt index offset size " << triangles.sbtIndexOffsetSizeInBytes;
            result = false;
        }
        if( triangles.sbtIndexOffsetStrideInBytes != 0 )
        {
            *separator() << "has non-zero sbt index offset stride " << triangles.sbtIndexOffsetStrideInBytes;
            result = false;
        }
        if( result )
        {
            *listener << "has no sbt index offsets";
        }
        return result;
    };
}

inline OptixTriangleBuildInputPredicate hasNoPrimitiveIndexOffset()
{
    return []( ::testing::MatchResultListener* listener, const OptixBuildInputTriangleArray& triangles ) {
        if( triangles.primitiveIndexOffset != 0 )
        {
            *listener << "has non-zero primitive index offset " << triangles.primitiveIndexOffset;
            return false;
        }

        *listener << "has zero primitive index offset";
        return true;
    };
}

inline OptixTriangleBuildInputPredicate hasNoOpacityMap()
{
    return []( ::testing::MatchResultListener* listener, const OptixBuildInputTriangleArray& triangles ) {
        bool result = true;
#if OPTIX_VERSION >= 70600
        auto separator = [&] {
            if( !result )
                *listener << "; ";
            return listener;
        };
        const OptixBuildInputOpacityMicromap& opacityMicromap = triangles.opacityMicromap;
        if( opacityMicromap.indexingMode != OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_NONE )
        {
            *listener << "has opacity micromap indexing mode " << opacityMicromap.indexingMode
                      << ", expected OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_NONE ("
                      << OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_NONE << ')';
            result = false;
        }
        if( opacityMicromap.opacityMicromapArray != CUdeviceptr{} )
        {
            *separator() << "has non-zero opacity micromap array " << opacityMicromap.opacityMicromapArray;
            result = false;
        }
        if( opacityMicromap.indexBuffer != CUdeviceptr{} )
        {
            *separator() << "has non-zero opacity micromap index buffer " << opacityMicromap.indexBuffer;
            result = false;
        }
        if( opacityMicromap.indexSizeInBytes != 0U )
        {
            *separator() << "has non-zero opacity micromap index size " << opacityMicromap.indexSizeInBytes;
            result = false;
        }
        if( opacityMicromap.indexStrideInBytes != 0U )
        {
            *separator() << "has non-zero opacity micromap index stride " << opacityMicromap.indexStrideInBytes;
            result = false;
        }
        if( opacityMicromap.indexOffset != 0U )
        {
            *separator() << "has non-zero opacity micromap index offset " << opacityMicromap.indexOffset;
            result = false;
        }
        if( opacityMicromap.numMicromapUsageCounts != 0U )
        {
            *separator() << "has non-zero opacity micromap usage count " << opacityMicromap.numMicromapUsageCounts;
            result = false;
        }
        if( opacityMicromap.micromapUsageCounts != nullptr )
        {
            *separator() << "has non-null opacity micromap usage count array " << opacityMicromap.micromapUsageCounts;
            result = false;
        }
#endif
        if( result )
        {
            *listener << "has no opacity micromap";
        }

        return result;
    };
}

MATCHER_P2( hasTriangleBuildInput, n, predicate, "" )
{
    if( arg[n].type != OPTIX_BUILD_INPUT_TYPE_TRIANGLES )
    {
        *result_listener << "input " << n << " is of type " << arg[n].type
                         << ", expected OPTIX_BUILD_INPUT_TYPE_TRIANGLES (" << OPTIX_BUILD_INPUT_TYPE_TRIANGLES << ')';
        return false;
    }

    *result_listener << "input " << n << " is of type OPTIX_BUILD_INPUT_TYPE_TRIANGLES ("
                     << OPTIX_BUILD_INPUT_TYPE_TRIANGLES << "); ";
    return predicate( result_listener, arg[n].triangleArray );
}

// OptixAccelBuildOptions
MATCHER( isBuildOperation, "" )
{
    if( arg->operation != OPTIX_BUILD_OPERATION_BUILD )
    {
        *result_listener << "build operation is " << arg->operation << ", expected OPTIX_BUILD_OPERATION_BUILD ("
                         << OPTIX_BUILD_OPERATION_BUILD << ')';
        return false;
    }

    *result_listener << "build operation is OPTIX_BUILD_OPERATION_BUILD (" << OPTIX_BUILD_OPERATION_BUILD << ')';
    return true;
}

// OptixAccelBuildOptions
MATCHER( buildAllowsUpdate, "" )
{
    if( ( arg->buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE ) == 0 )
    {
        *result_listener << "build flag OPTIX_BUILD_FLAG_ALLOW_UPDATE (" << OPTIX_BUILD_FLAG_ALLOW_UPDATE
                         << ") not set in value " << arg->buildFlags;
        return false;
    }

    *result_listener << "build flag OPTIX_BUILD_FLAG_ALLOW_UPDATE (" << OPTIX_BUILD_FLAG_ALLOW_UPDATE
                     << ") set in value " << arg->buildFlags;
    return true;
}

// OptixAccelBuildOptions
MATCHER( buildAllowsRandomVertexAccess, "" )
{
    if( ( arg->buildFlags & OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS ) == 0 )
    {
        *result_listener << "build flag OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS ("
                         << OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS << ") not set in value " << arg->buildFlags;
        return false;
    }
    *result_listener << "build flag OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS ("
                     << OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS << ") set in value " << arg->buildFlags;
    return true;
}

inline OptixCustomPrimitiveBuildInputPredicate hasNumCustomPrimitives( unsigned int numPrims )
{
    return [=]( ::testing::MatchResultListener* listener, const OptixBuildInputCustomPrimitiveArray& prims ) {
        if( prims.numPrimitives != numPrims )
        {
            *listener << "has " << prims.numPrimitives << " primitives, expected " << numPrims;
            return false;
        }

        *listener << "has " << numPrims << " primitives";
        return true;
    };
}

MATCHER_P2( hasCustomPrimitiveBuildInput, n, predicate, "" )
{
    if( arg[n].type != OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES )
    {
        *result_listener << "input " << n << " is of type " << arg[n].type << ", not OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES ("
                         << OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES << ')';
        return false;
    }

    *result_listener << "input " << n << " is of type OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES ("
                     << OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES << "); ";
    return predicate( result_listener, arg[n].customPrimitiveArray );
}

inline OptixSphereBuildInputPredicate hasSphereVertexStride( unsigned int value )
{
    return [=]( ::testing::MatchResultListener* listener, const OptixBuildInputSphereArray& spheres ) {
        return hasEqualValues( listener, "sphere vertex stride", value, spheres.vertexStrideInBytes );
    };
}

inline OptixSphereBuildInputPredicate hasSphereRadiusStride( unsigned int value )
{
    return [=]( ::testing::MatchResultListener* listener, const OptixBuildInputSphereArray& spheres ) {
        return hasEqualValues( listener, "sphere radius stride", value, spheres.radiusStrideInBytes );
    };
}

inline OptixSphereBuildInputPredicate hasSphereSingleRadius( bool value )
{
    return [=]( ::testing::MatchResultListener* listener, const OptixBuildInputSphereArray& spheres ) {
        return hasEqualValues( listener, "sphere single radius", value ? 1 : 0, spheres.singleRadius );
    };
}

inline OptixSphereBuildInputPredicate hasSphereFlags( const std::vector<unsigned int>& flags )
{
    return [&]( ::testing::MatchResultListener* listener, const OptixBuildInputSphereArray& spheres ) {
        if( spheres.flags == nullptr )
        {
            *listener << "has null flags array";
            return false;
        }

        bool result{ true };
        auto separator = [&] {
            if( !result )
                *listener << "; ";
            return listener;
        };
        if( spheres.numSbtRecords != flags.size() )
        {
            *separator() << "has num SBT records " << spheres.numSbtRecords << ", expected " << flags.size();
            result = false;
        }
        for( unsigned int i = 0; i < spheres.numSbtRecords; ++i )
        {
            if( i == flags.size() )
            {
                break;
            }

            if( spheres.flags[i] != flags[i] )
            {
                *separator() << "has flags[" << i << "] " << spheres.flags[i] << ", expected " << flags[i];
            }
        }
        return result;
    };
}

inline OptixSphereBuildInputPredicate hasSphereSbtRecordCount( unsigned int value )
{
    return [=]( ::testing::MatchResultListener* listener, const OptixBuildInputSphereArray& spheres ) {
        return hasEqualValues( listener, "SBT record count", value, spheres.numSbtRecords );
    };
}

inline OptixSphereBuildInputPredicate hasNoSphereSbtIndexOffsets()
{
    return []( ::testing::MatchResultListener* listener, const OptixBuildInputSphereArray& spheres ) {
        const bool nullIndexBuffer =
            hasEqualValues( listener, "SBT index offset buffer", CUdeviceptr{}, spheres.sbtIndexOffsetBuffer );
        const bool zeroOffsetSize = hasEqualValues( listener, "SBT index offset size", 0U, spheres.sbtIndexOffsetSizeInBytes );
        const bool zeroOffsetStride = hasEqualValues( listener, "SBT index offset stride", 0U, spheres.sbtIndexOffsetStrideInBytes );
        return nullIndexBuffer && zeroOffsetSize && zeroOffsetStride;
    };
}

inline OptixSphereBuildInputPredicate hasSpherePrimitiveIndexOffset( unsigned int value )
{
    return [=]( ::testing::MatchResultListener* listener, const OptixBuildInputSphereArray& spheres ) {
        return hasEqualValues( listener, "primitive index offset", value, spheres.primitiveIndexOffset );
    };
}

inline ListenerPredicate<OptixBuildInputSphereArray> hasDeviceSphereSingleRadius( float expectedSingleRadius )
{
    return [=]( ::testing::MatchResultListener* listener, const OptixBuildInputSphereArray& spheres ) {
        float actualSingleRadius{};
        OTK_ERROR_CHECK( cudaMemcpy( &actualSingleRadius, otk::bit_cast<void*>( spheres.radiusBuffers[0] ),
                                     sizeof( float ), cudaMemcpyDeviceToHost ) );
        if( actualSingleRadius != expectedSingleRadius )
        {
            *listener << "has radius " << actualSingleRadius << ", expected " << expectedSingleRadius;
            return false;
        }

        *listener << "has radius " << expectedSingleRadius;
        return true;
    };
}

MATCHER_P2( hasSphereBuildInput, n, predicate, "" )
{
    if( arg[n].type != OPTIX_BUILD_INPUT_TYPE_SPHERES )
    {
        *result_listener << "input " << n << " is of type " << toString( arg[n].type ) << " (" << arg[n].type
                         << "), expected OPTIX_BUILD_INPUT_TYPE_SPHERES (" << OPTIX_BUILD_INPUT_TYPE_SPHERES << ')';
        return false;
    }
    *result_listener << "input " << n << " is of type OPTIX_BUILD_INPUT_TYPE_SPHERES (" << OPTIX_BUILD_INPUT_TYPE_SPHERES << ")";

    bool                              result{ true };
    const OptixBuildInputSphereArray& spheres = arg[n].sphereArray;
    if( spheres.vertexBuffers == nullptr )
    {
        *result_listener << "; has null vertex buffer array";
        result = false;
    }
    if( spheres.vertexBuffers != nullptr && spheres.vertexBuffers[0] == CUdeviceptr{} )
    {
        *result_listener << "has null vertex buffer 0";
        result = false;
    }
    if( spheres.radiusBuffers == nullptr )
    {
        *result_listener << "has null radius buffer array";
        result = false;
    }
    if( spheres.radiusBuffers != nullptr && spheres.radiusBuffers[0] == CUdeviceptr{} )
    {
        *result_listener << "has null radius buffer 0";
        result = false;
    }
    const bool predResult = predicate( result_listener, arg[n].sphereArray );
    return result && predResult;
}

/// Assertion for testing instance 4x3 transforms, e.g. ASSERT_TRUE( isSameTransform( expected, actual ) )
inline ::testing::AssertionResult isSameTransform( const float ( &expectedTransform )[12], const float ( &transform )[12] )
{
    auto compare = []( float lhs, float rhs ) { return std::abs( rhs - lhs ) < 1.0e-6f; };
    if( std::equal( std::begin( expectedTransform ), std::end( expectedTransform ), std::begin( transform ), compare ) )
        return ::testing::AssertionSuccess() << "transforms are equal";

    return ::testing::AssertionFailure() << getRangeDifferences( std::begin( expectedTransform ), std::end( expectedTransform ),
                                                                 std::begin( transform ), compare );
}

namespace detail {

inline bool programGroupDescsContain( const OptixProgramGroupDesc* begin, int numDescs, const OptixProgramGroupDesc& desc )
{
    const OptixProgramGroupDesc* end = begin + numDescs;
    return std::find( begin, end, desc ) != end;
}

inline const char* nameOrNullPtr( const char* name )
{
    return name == nullptr ? "(nullptr)" : name;
}

}  // namespace detail

MATCHER_P3( hasRayGenDesc, count, module, entryPoint, "" )
{
    OptixProgramGroupDesc desc{ OPTIX_PROGRAM_GROUP_KIND_RAYGEN, OPTIX_PROGRAM_GROUP_FLAGS_NONE, { module, entryPoint } };
    const bool result = detail::programGroupDescsContain( arg, count, desc );
    if( !result )
    {
        *result_listener << "raygen group desc (" << module << ", " << detail::nameOrNullPtr( entryPoint )
                         << ") not found in descs[" << count << ']';
    }
    else
    {
        *result_listener << "raygen group desc (" << module << ", " << detail::nameOrNullPtr( entryPoint )
                         << ") found in descs[" << count << ']';
    }
    return result;
}

MATCHER_P3( hasMissDesc, count, module, entryPoint, "" )
{
    OptixProgramGroupDesc desc{ OPTIX_PROGRAM_GROUP_KIND_MISS, OPTIX_PROGRAM_GROUP_FLAGS_NONE, { module, entryPoint } };
    const bool            result = detail::programGroupDescsContain( arg, count, desc );
    if( !result )
    {
        *result_listener << "miss group desc (" << module << ", " << detail::nameOrNullPtr( entryPoint )
                         << ") not found in descs[" << count << ']';
    }
    else
    {
        *result_listener << "miss group desc (" << module << ", " << detail::nameOrNullPtr( entryPoint )
                         << ") found in descs[" << count << ']';
    }
    return result;
}

MATCHER_P5( hasHitGroupISCHDesc, count, isModule, isEntryPoint, chModule, chEntryPoint, "" )
{
    OptixProgramGroupDesc      desc{ OPTIX_PROGRAM_GROUP_KIND_HITGROUP, OPTIX_PROGRAM_GROUP_FLAGS_NONE, { { 0, 0 } } };
    OptixProgramGroupHitgroup& hitgroup = desc.hitgroup;
    hitgroup.moduleIS                   = isModule;
    hitgroup.entryFunctionNameIS        = isEntryPoint;
    hitgroup.moduleAH                   = nullptr;
    hitgroup.entryFunctionNameAH        = nullptr;
    hitgroup.moduleCH                   = chModule;
    hitgroup.entryFunctionNameCH        = chEntryPoint;
    const bool result                   = detail::programGroupDescsContain( arg, count, desc );
    if( !result )
    {
        *result_listener << "hitgroup desc (IS(" << isModule << ", " << detail::nameOrNullPtr( isEntryPoint ) << "), CH(" << chModule
                         << ", " << detail::nameOrNullPtr( chEntryPoint ) << ")) not found in descs[" << count << ']';
    }
    else
    {
        *result_listener << "hitgroup desc (IS(" << isModule << ", " << detail::nameOrNullPtr( isEntryPoint ) << "), CH("
                         << chModule << ", " << detail::nameOrNullPtr( chEntryPoint ) << ")) found in descs[" << count << ']';
    }
    return result;
}

MATCHER_P7( hasHitGroupISAHCHDesc, count, isModule, isEntryPoint, ahModule, ahEntryPoint, chModule, chEntryPoint, "" )
{
    OptixProgramGroupDesc      desc{ OPTIX_PROGRAM_GROUP_KIND_HITGROUP, OPTIX_PROGRAM_GROUP_FLAGS_NONE, { { 0, 0 } } };
    OptixProgramGroupHitgroup& hitgroup = desc.hitgroup;
    hitgroup.moduleIS                   = isModule;
    hitgroup.entryFunctionNameIS        = isEntryPoint;
    hitgroup.moduleAH                   = ahModule;
    hitgroup.entryFunctionNameAH        = ahEntryPoint;
    hitgroup.moduleCH                   = chModule;
    hitgroup.entryFunctionNameCH        = chEntryPoint;
    const bool result                   = detail::programGroupDescsContain( arg, count, desc );
    if( !result )
    {
        *result_listener << "hitgroup desc (IS(" << isModule << ", " << detail::nameOrNullPtr( isEntryPoint ) << "), AH("
                         << ahModule << ", " << detail::nameOrNullPtr( ahEntryPoint ) << "), CH(" << chModule << ", "
                         << detail::nameOrNullPtr( chEntryPoint ) << ")) not found in descs[" << count << ']';
    }
    else
    {
        *result_listener << "hitgroup desc (IS(" << isModule << ", " << detail::nameOrNullPtr( isEntryPoint ) << "), AH("
                         << ahModule << ", " << detail::nameOrNullPtr( ahEntryPoint ) << "), CH(" << chModule << ", "
                         << detail::nameOrNullPtr( chEntryPoint ) << ")) found in descs[" << count << ']';
    }
    return result;
}

// OptixPipelineCompileOptions
MATCHER( usesMotionBlur, "" )
{
    if( arg->usesMotionBlur == 0 )
    {
        *result_listener << "motion blur is off";
        return false;
    }

    *result_listener << "motion blur is on";
    return true;
}

// OptixPipelineCompileOptions
MATCHER( allowAnyTraversableGraph, "" )
{
    if( arg->traversableGraphFlags != 0 )
    {
        *result_listener << "has traversable graph flags " << arg->traversableGraphFlags
                         << ", expected OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY ("
                         << OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY << ")";
        return false;
    }

    *result_listener << "has traversable graph flags OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY ("
                     << OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY << ")";
    return true;
}

// OptixPipelineCompileOptions
MATCHER_P( hasPayloadValueCount, n, "" )
{
    if( arg->numPayloadValues != n )
    {
        *result_listener << "expected " << n << " payload value" << ( n > 1 ? "s" : "" ) << ", got " << arg->numPayloadValues;
        return false;
    }

    *result_listener << "has " << n << " payload value" << ( n > 1 ? "s" : "" );
    return true;
}

// OptixPipelineCompileOptions
MATCHER_P( hasAttributeValueCount, n, "" )
{
    if( arg->numAttributeValues != n )
    {
        *result_listener << "expected " << n << " attribute value" << ( n > 1 ? "s" : "" ) << ", got " << arg->numAttributeValues;
        return false;
    }

    *result_listener << "has " << n << " attribute value" << ( n > 1 ? "s" : "" );
    return true;
}

// OptixPipelineCompileOptions
MATCHER_P( hasExceptionFlags, flags, "" )
{
    if( arg->exceptionFlags != static_cast<unsigned int>( flags ) )
    {
        *result_listener << "expected " << flags << " (0x" << std::hex << flags << ") exception flags, got " << std::dec
                         << arg->exceptionFlags << " (0x" << std::hex << arg->exceptionFlags << ")";
        return false;
    }

    *result_listener << "has " << std::dec << arg->exceptionFlags << " (0x" << std::hex << arg->exceptionFlags << ") exception flags";
    return true;
}

// OptixPipelineCompileOptions
MATCHER_P( hasParamsName, name, "" )
{
    if( arg->pipelineLaunchParamsVariableName == nullptr )
    {
        *result_listener << "params variable name is nullptr";
        return false;
    }
    if( std::string( arg->pipelineLaunchParamsVariableName ) != name )
    {
        *result_listener << "expected params variable name '" << name << "', got '" << arg->pipelineLaunchParamsVariableName << "'";
        return false;
    }

    *result_listener << "has params variable name '" << name << "'";
    return true;
}

// OptixPipelineCompileOptions
MATCHER_P( hasPrimitiveTypes, primitives, "" )
{
    if( arg->usesPrimitiveTypeFlags != static_cast<unsigned int>( primitives ) )
    {
        *result_listener << "expected primitive types " << primitives << " (0x" << std::hex << primitives << "), got "
                         << std::dec << arg->usesPrimitiveTypeFlags;
        return false;
    }

    *result_listener << "has params primitive types " << primitives << " (0x" << std::hex << primitives << ")";
    return true;
}

#if OPTIX_VERSION >= 70600
// OptixPipelineCompileOptions
MATCHER( allowsOpacityMicromaps, "" )
{
    if( arg->allowOpacityMicromaps == 0 )
    {
        *result_listener << "opacity micromaps are disallowed";
        return false;
    }

    *result_listener << "opacity micromaps are allowed";
    return true;
}
#endif

// OptixPipelineLinkOptions
MATCHER_P( hasMaxTraceDepth, n, "" )
{
    if( arg->maxTraceDepth != n )
    {
        *result_listener << "expected max trace depth " << n << ", got " << arg->maxTraceDepth;
        return false;
    }
    *result_listener << "has max trace depth " << n;
    return true;
}

}  // namespace testing
}  // namespace otk
