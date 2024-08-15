// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <optix.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

/// Comparison operators and stream insertion operators for OptiX structures.  This enables
/// gtest matchers to compare the values of structures and output their values when they don't
/// match.

namespace otk {
namespace testing {
namespace detail {

inline bool stringsBothNullOrSame( const char* lhs, const char* rhs )
{
    return ( lhs == nullptr && rhs == nullptr ) || ( lhs != nullptr && rhs != nullptr && std::string{ lhs } == rhs );
}

inline void stringOrNullPtr( std::ostream& str, const char* value )
{
    str << ( value != nullptr ? value : "nullptr" );
}

inline void flags( std::ostream& str, unsigned int value )
{
    const char fill = str.fill();
    str << value << " (0x" << std::hex << std::nouppercase << std::setfill( '0' )
        << std::setw( sizeof( unsigned int ) * 2 ) << value << std::dec << std::setw( 0 ) << std::setfill( fill ) << ")";
}

}  // namespace detail
}  // namespace testing
}  // namespace otk

// To satisfy argument-dependent lookup, the comparison operators have to be in the global scope,
// where OptiX structures are declared.

inline bool operator==( const OptixAabb& lhs, const OptixAabb& rhs )
{
    // clang-format off
    return
        lhs.minX == rhs.minX && lhs.minY == rhs.minY && lhs.minZ == rhs.minZ &&
        lhs.maxX == rhs.maxX && lhs.maxY == rhs.maxY && lhs.maxZ == rhs.maxZ;
    // clang-format on
}

inline bool operator!=( const OptixAabb& lhs, const OptixAabb& rhs )
{
    return !( lhs == rhs );
}

inline std::ostream& operator<<( std::ostream& str, const OptixAabb& val )
{
    return str << "{ min(" << val.minX << ", " << val.minY << ", " << val.minZ << "), max(" << val.maxX << ", "
               << val.maxY << ", " << val.maxZ << ") }";
}

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

inline bool operator==( const OptixInstance& lhs, const OptixInstance& rhs )
{
    // clang-format off
    return std::equal( std::begin( lhs.transform ), std::end( lhs.transform ), std::begin( rhs.transform ) )
        && lhs.instanceId        == rhs.instanceId
        && lhs.sbtOffset         == rhs.sbtOffset
        && lhs.visibilityMask    == rhs.visibilityMask
        && lhs.flags             == rhs.flags
        && lhs.traversableHandle == rhs.traversableHandle;
    // clang-format on
}

inline bool operator!=( const OptixInstance& lhs, const OptixInstance& rhs )
{
    return !( lhs == rhs );
}

inline std::ostream& operator<<( std::ostream& str, const OptixInstance& val )
{
    str << "Instance{ [ ";
    bool first = true;
    for( int i = 0; i < 12; ++i )
    {
        if( !first )
            str << ", ";
        first = false;
        str << val.transform[i];
    }
    return str << " ], " << val.instanceId << ", " << val.sbtOffset << ", " << val.visibilityMask << ", " << val.flags
               << ", " << val.traversableHandle << " }";
}

inline bool operator==( const OptixPipelineCompileOptions& lhs, const OptixPipelineCompileOptions& rhs )
{
    // clang-format off
    return
        lhs.usesMotionBlur          == rhs.usesMotionBlur &&
        lhs.traversableGraphFlags   == rhs.traversableGraphFlags &&
        lhs.numPayloadValues        == rhs.numPayloadValues &&
        lhs.numAttributeValues      == rhs.numAttributeValues &&
        lhs.exceptionFlags          == rhs.exceptionFlags &&
        otk::testing::detail::stringsBothNullOrSame( lhs.pipelineLaunchParamsVariableName, rhs.pipelineLaunchParamsVariableName ) &&
        lhs.usesPrimitiveTypeFlags  == rhs.usesPrimitiveTypeFlags &&
#if OPTIX_VERSION >= 70600
        lhs.allowOpacityMicromaps   == rhs.allowOpacityMicromaps;
#else
        true;
#endif
    // clang-format on
}

inline bool operator!=( const OptixPipelineCompileOptions& lhs, const OptixPipelineCompileOptions& rhs )
{
    return !( lhs == rhs );
}

inline std::ostream& operator<<( std::ostream& str, const OptixPipelineCompileOptions& value )
{
    str << "PipelineCompileOptions{ " << value.usesMotionBlur << ", ";
    otk::testing::detail::flags( str, value.traversableGraphFlags );
    str << ", " << value.numPayloadValues << ", " << value.numAttributeValues << ", ";
    otk::testing::detail::flags( str, value.exceptionFlags );
    str << ", ";
    otk::testing::detail::stringOrNullPtr( str, value.pipelineLaunchParamsVariableName );
    str << ", ";
    otk::testing::detail::flags( str, value.usesPrimitiveTypeFlags );
#if OPTIX_VERSION >= 70600
    str << ", " << value.allowOpacityMicromaps;
#endif
    return str << " }";
}

inline const char* toString( OptixProgramGroupKind kind )
{
    switch( kind )
    {
        case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
            return "OPTIX_PROGRAM_GROUP_KIND_RAYGEN";
        case OPTIX_PROGRAM_GROUP_KIND_MISS:
            return "OPTIX_PROGRAM_GROUP_KIND_MISS";
        case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
            return "OPTIX_PROGRAM_GROUP_KIND_EXCEPTION";
        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
            return "OPTIX_PROGRAM_GROUP_KIND_HITGROUP";
        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
            return "OPTIX_PROGRAM_GROUP_KIND_CALLABLES";
    }
    return "";
}

// Printing out what amounts to a void* is platform specific.  One some platforms you
// get uppercase hex digits on some platforms you get lowercase hex digits.  For OptixModule
// define a stream insertion operator that converts the opaque pointer to an unsigned long long
// and prints that as a hexadecimal value.
inline std::ostream& operator<<( std::ostream& str, OptixModule value )
{
    static_assert( sizeof( OptixModule ) == sizeof( unsigned long long ),
                   "OptixModule is not the same size as unsigned long long" );
    unsigned long long val{};
    std::memcpy( &val, &value, sizeof( unsigned long long ) );
    char fill = str.fill();
    return str << "0x" << std::nouppercase << std::setw( sizeof( unsigned long long ) * 2 ) << std::setfill( '0' )
               << std::hex << val << std::setfill( fill );
}

namespace otk {
namespace testing {
namespace detail {

inline std::ostream& entryPoint( std::ostream& str, const char* prefix, OptixModule mod, const char* entryFunctionName )
{
    str << prefix << "{ " << mod << ", ";
    stringOrNullPtr( str, entryFunctionName );
    return str << " }";
}

}  // namespace detail
}  // namespace testing
}  // namespace otk

inline std::ostream& operator<<( std::ostream& str, const OptixProgramGroupSingleModule& val )
{
    return otk::testing::detail::entryPoint( str, "SingleModule", val.module, val.entryFunctionName );
}

inline std::ostream& operator<<( std::ostream& str, const OptixProgramGroupHitgroup& val )
{
    str << "HitGroup{ ";
    otk::testing::detail::entryPoint( str, "IS", val.moduleIS, val.entryFunctionNameIS ) << ", ";
    otk::testing::detail::entryPoint( str, "AH", val.moduleAH, val.entryFunctionNameAH ) << ", ";
    return otk::testing::detail::entryPoint( str, "CH", val.moduleCH, val.entryFunctionNameCH ) << " }";
}

inline std::ostream& operator<<( std::ostream& str, const OptixProgramGroupCallables& val )
{
    str << "Callables{ ";
    otk::testing::detail::entryPoint( str, "DC", val.moduleDC, val.entryFunctionNameDC ) << ", ";
    return otk::testing::detail::entryPoint( str, "CC", val.moduleCC, val.entryFunctionNameCC ) << " }";
}

inline std::ostream& operator<<( std::ostream& str, const OptixProgramGroupDesc& val )
{
    str << "ProgramGroup{ " << toString( val.kind ) << " (" << val.kind << "), ";
    switch( val.kind )
    {
        case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
            str << val.raygen;
            break;
        case OPTIX_PROGRAM_GROUP_KIND_MISS:
            str << val.miss;
            break;
        case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
            str << val.exception;
            break;
        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
            str << val.hitgroup;
            break;
        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
            str << val.callables;
            break;
    }
    return str << " }";
}
