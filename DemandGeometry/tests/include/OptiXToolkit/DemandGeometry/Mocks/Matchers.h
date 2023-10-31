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

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Memory/BitCast.h>

#include <optix.h>

#include <gmock/gmock.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>

namespace otk {
namespace testing {
namespace detail {

inline bool stringsBothNullOrSame( const char* lhs, const char* rhs )
{
    return ( lhs == nullptr && rhs == nullptr ) || ( lhs != nullptr && rhs != nullptr && std::string{ lhs } == rhs );
}

}  // namespace detail
}  // namespace testing
}  // namespace otk

inline bool operator==( const OptixProgramGroupSingleModule& lhs, const OptixProgramGroupSingleModule& rhs )
{
    return lhs.module == rhs.module && otk::testing::detail::stringsBothNullOrSame( lhs.entryFunctionName, rhs.entryFunctionName );
}

inline bool operator!=( const OptixProgramGroupSingleModule& lhs, const OptixProgramGroupSingleModule& rhs )
{
    return !( lhs == rhs );
}

inline bool operator==( const OptixProgramGroupHitgroup& lhs, const OptixProgramGroupHitgroup& rhs )
{
    using namespace otk::testing::detail;
    return lhs.moduleCH == rhs.moduleCH && stringsBothNullOrSame( lhs.entryFunctionNameCH, rhs.entryFunctionNameCH )
           && lhs.moduleAH == rhs.moduleAH && stringsBothNullOrSame( lhs.entryFunctionNameAH, rhs.entryFunctionNameAH )
           && lhs.moduleIS == rhs.moduleIS && stringsBothNullOrSame( lhs.entryFunctionNameIS, rhs.entryFunctionNameIS );
}

inline bool operator!=( const OptixProgramGroupHitgroup& lhs, const OptixProgramGroupHitgroup& rhs )
{
    return !( lhs == rhs );
}

inline bool operator==( const OptixProgramGroupCallables& lhs, const OptixProgramGroupCallables& rhs )
{
    using namespace otk::testing::detail;
    return lhs.moduleDC == rhs.moduleDC && stringsBothNullOrSame( lhs.entryFunctionNameDC, rhs.entryFunctionNameDC )
           && lhs.moduleCC == rhs.moduleCC && stringsBothNullOrSame( lhs.entryFunctionNameCC, rhs.entryFunctionNameCC );
}

inline bool operator!=( const OptixProgramGroupCallables& lhs, const OptixProgramGroupCallables& rhs )
{
    return !( lhs == rhs );
}

inline bool operator==( const OptixProgramGroupDesc& lhs, const OptixProgramGroupDesc& rhs )
{
    if( lhs.kind != rhs.kind || lhs.flags != rhs.flags )
        return false;

    switch( lhs.kind )
    {
        case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
            return lhs.raygen == rhs.raygen;
        case OPTIX_PROGRAM_GROUP_KIND_MISS:
            return lhs.miss == rhs.miss;
        case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
            return lhs.exception == rhs.exception;
        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
            return lhs.hitgroup == rhs.hitgroup;
        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
            return lhs.callables == rhs.callables;
    }

    // No kind and no flags are equal, regardless of union contents.
    return lhs.kind == 0 && lhs.flags == 0;
}

inline bool operator!=( const OptixProgramGroupDesc& lhs, const OptixProgramGroupDesc& rhs )
{
    return !( lhs == rhs );
}

namespace otk {
namespace testing {

MATCHER_P( isInstanceBuildInput, n, "" )
{
    if( arg[n].type != OPTIX_BUILD_INPUT_TYPE_INSTANCES )
    {
        *result_listener << "input " << n << " is of type " << arg[n].type
                         << ", expected OPTIX_BUILD_INPUT_TYPE_INSTANCES (" << OPTIX_BUILD_INPUT_TYPE_INSTANCES << ')';
        return false;
    }
    return true;
}

MATCHER_P( isTriangleBuildInput, n, "" )
{
    if( arg[n].type != OPTIX_BUILD_INPUT_TYPE_TRIANGLES )
    {
        *result_listener << "input " << n << " is of type " << arg[n].type
                         << ", expected OPTIX_BUILD_INPUT_TYPE_TRIANGLES (" << OPTIX_BUILD_INPUT_TYPE_TRIANGLES << ')';
        return false;
    }
    return true;
}

MATCHER( isBuildOperation, "" )
{
    if( arg->operation != OPTIX_BUILD_OPERATION_BUILD )
    {
        *result_listener << "build operation is " << arg->operation << ", expected OPTIX_BUILD_OPERATION_BUILD ("
                         << OPTIX_BUILD_OPERATION_BUILD << ')';
        return false;
    }
    return true;
}

MATCHER( buildAllowsUpdate, "" )
{
    if( ( arg->buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE ) == 0 )
    {
        *result_listener << "build flag OPTIX_BUILD_FLAG_ALLOW_UPDATE (" << OPTIX_BUILD_FLAG_ALLOW_UPDATE
                         << ") not set in value " << arg->buildFlags;
        return false;
    }
    return true;
}

MATCHER( buildAllowsRandomVertexAccess, "" )
{
    if( ( arg->buildFlags & OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS ) == 0 )
    {
        *result_listener << "build flag OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS ("
                         << OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS << ") not set in value " << arg->buildFlags;
        return false;
    }
    return true;
}

MATCHER_P( isCustomPrimitiveBuildInput, n, "" )
{
    if( arg[n].type != OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES )
    {
        *result_listener << "input " << n << " is of type " << arg[n].type << ", not OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES ("
                         << OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES << ')';
        return false;
    }
    return true;
}

MATCHER_P( isZeroInstances, n, "" )
{
    if( arg[n].instanceArray.numInstances != 0 )
    {
        *result_listener << "input " << n << " has non-zero numInstances (" << arg[n].instanceArray.numInstances << ')';
        return false;
    }
    return true;
}

MATCHER_P2( hasNumInstances, n, num, "" )
{
    if( arg[n].type != OPTIX_BUILD_INPUT_TYPE_INSTANCES )
    {
        *result_listener << "input " << n << " is of type " << arg[n].type
                         << ", expected OPTIX_BUILD_INPUT_TYPE_INSTANCES (" << OPTIX_BUILD_INPUT_TYPE_INSTANCES << ')';
        return false;
    }
    const OptixBuildInputInstanceArray& instances = arg[n].instanceArray;
    if( num != instances.numInstances )
    {
        *result_listener << "input " << n << " has " << instances.numInstances << " instances, expected " << num;
        return false;
    }

    return true;
}

MATCHER_P3( hasDeviceInstanceId, n, index, id, "" )
{
    if( arg[n].type != OPTIX_BUILD_INPUT_TYPE_INSTANCES )
    {
        *result_listener << "input " << n << " is of type " << arg[n].type
                         << ", expected OPTIX_BUILD_INPUT_TYPE_INSTANCES (" << OPTIX_BUILD_INPUT_TYPE_INSTANCES << ')';
        return false;
    }
    const OptixBuildInputInstanceArray& instances = arg[n].instanceArray;
    if( index >= instances.numInstances )
    {
        *result_listener << "input " << n << " instance index " << index << " exceeds " << instances.numInstances;
        return false;
    }
    std::vector<OptixInstance> actualInstances;
    actualInstances.resize( instances.numInstances );
    OTK_ERROR_CHECK( cudaMemcpy( actualInstances.data(), otk::bit_cast<void*>( instances.instances ),
                                 sizeof( OptixInstance ) * instances.numInstances, cudaMemcpyDeviceToHost ) );
    if( actualInstances[index].instanceId != id )
    {
        *result_listener << "input " << n << " instance " << index << " has different id "
                         << actualInstances[index].instanceId << " != " << id;
        return false;
    }

    return true;
}

inline std::string compareRanges( const float* begin, const float* end, const float* rhs, const std::function<bool( float, float )>& compare )
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

/// Assertion for testing instance 4x3 transforms, e.g. ASSERT_TRUE( isSameeTransfrom( expected, actual ) )
inline ::testing::AssertionResult isSameTransform( const float ( &expectedTransform )[12], const float ( &transform )[12] )
{
    auto compare = []( float lhs, float rhs ) { return std::abs( rhs - lhs ) < 1.0e-6f; };
    if( std::equal( std::begin( expectedTransform ), std::end( expectedTransform ), std::begin( transform ), compare ) )
        return ::testing::AssertionSuccess() << "transforms are equal";

    return ::testing::AssertionFailure()
           << compareRanges( std::begin( expectedTransform ), std::end( expectedTransform ), std::begin( transform ), compare );
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
    OptixProgramGroupDesc desc{ OPTIX_PROGRAM_GROUP_KIND_RAYGEN, OPTIX_PROGRAM_GROUP_FLAGS_NONE };
    desc.raygen       = { module, entryPoint };
    const bool result = detail::programGroupDescsContain( arg, count, desc );
    if( !result )
    {
        *result_listener << "raygen group desc (" << module << ", " << detail::nameOrNullPtr( entryPoint )
                         << ") not found in descs[" << count << ']';
    }
    return result;
}

MATCHER_P3( hasMissDesc, count, module, entryPoint, "" )
{
    OptixProgramGroupDesc desc{ OPTIX_PROGRAM_GROUP_KIND_MISS, OPTIX_PROGRAM_GROUP_FLAGS_NONE };
    desc.miss         = { module, entryPoint };
    const bool result = detail::programGroupDescsContain( arg, count, desc );
    if( !result )
    {
        *result_listener << "miss group desc (" << module << ", " << detail::nameOrNullPtr( entryPoint )
                         << ") not found in descs[" << count << ']';
    }
    return result;
}

MATCHER_P5( hasHitGroupISCHDesc, count, isModule, isEntryPoint, chModule, chEntryPoint, "" )
{
    OptixProgramGroupDesc      desc{ OPTIX_PROGRAM_GROUP_KIND_HITGROUP, OPTIX_PROGRAM_GROUP_FLAGS_NONE };
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
        *result_listener << "hitgroup desc (" << chModule << ", " << detail::nameOrNullPtr( chEntryPoint ) << ", " << isModule
                         << ", " << detail::nameOrNullPtr( isEntryPoint ) << ") not found in descs[" << count << ']';
    }
    return result;
}

MATCHER_P7( hasHitGroupISAHCHDesc, count, isModule, isEntryPoint, ahModule, ahEntryPoint, chModule, chEntryPoint, "" )
{
    OptixProgramGroupDesc      desc{ OPTIX_PROGRAM_GROUP_KIND_HITGROUP, OPTIX_PROGRAM_GROUP_FLAGS_NONE };
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
        *result_listener << "hitgroup desc (" << chModule << ", " << detail::nameOrNullPtr( chEntryPoint ) << ", " << isModule
                         << ", " << detail::nameOrNullPtr( isEntryPoint ) << ") not found in descs[" << count << ']';
    }
    return result;
}

}  // namespace testing
}  // namespace otk
