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

#include <optix.h>

#include <gmock/gmock.h>

#include <algorithm>
#include <string>

namespace otk {
namespace testing {
namespace detail {

inline bool stringsBothNullOrSame( const char* lhs, const char* rhs )
{
    return ( lhs == nullptr && rhs == nullptr ) || std::string{ lhs } == rhs;
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
            lhs.exception == rhs.exception;
        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
            return lhs.hitgroup == rhs.hitgroup;
        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
            return lhs.callables == rhs.callables;
    }

    return false;
}

inline bool operator!=( const OptixProgramGroupDesc& lhs, const OptixProgramGroupDesc& rhs )
{
    return !( lhs == rhs );
}

namespace otk {
namespace testing {

MATCHER( isInstanceBuildInput, "" )
{
    return arg->type == OPTIX_BUILD_INPUT_TYPE_INSTANCES;
}

namespace detail {

inline bool programGroupDescsContain( const OptixProgramGroupDesc* begin, int numDescs, const OptixProgramGroupDesc& desc )
{
    const OptixProgramGroupDesc* end = begin + numDescs;
    return std::find( begin, end, desc ) != end;
}

}  // namespace detail

MATCHER_P3( hasRayGenDesc, count, module, entryPoint, "" )
{
    OptixProgramGroupDesc desc{ OPTIX_PROGRAM_GROUP_KIND_RAYGEN, OPTIX_PROGRAM_GROUP_FLAGS_NONE };
    desc.raygen = { module, entryPoint };
    return detail::programGroupDescsContain( arg, count, desc );
}

MATCHER_P3( hasMissDesc, count, module, entryPoint, "" )
{
    OptixProgramGroupDesc desc{ OPTIX_PROGRAM_GROUP_KIND_MISS, OPTIX_PROGRAM_GROUP_FLAGS_NONE };
    desc.miss = { module, entryPoint };
    return detail::programGroupDescsContain( arg, count, desc );
}

MATCHER_P7( hasHitGroupDesc, count, chModule, chEntryPoint, ahModule, ahEntryPoint, isModule, isEntryPoint, "" )
{
    OptixProgramGroupDesc      desc{ OPTIX_PROGRAM_GROUP_KIND_HITGROUP, OPTIX_PROGRAM_GROUP_FLAGS_NONE };
    OptixProgramGroupHitgroup& hitgroup = desc.hitgroup;
    hitgroup.moduleCH                   = chModule;
    hitgroup.entryFunctionNameCH        = chEntryPoint;
    hitgroup.moduleAH                   = ahModule;
    hitgroup.entryFunctionNameAH        = ahEntryPoint;
    hitgroup.moduleIS                   = isModule;
    hitgroup.entryFunctionNameIS        = isEntryPoint;
    return detail::programGroupDescsContain( arg, count, desc );
}

}  // namespace testing
}  // namespace otk
