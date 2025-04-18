// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <DemandPbrtScene/Options.h>
#include <DemandPbrtScene/UserInterfaceStatistics.h>

#include <ostream>
#include <string>

namespace demandPbrtScene {

template <typename Stats>
struct Json
{
    Json( const Stats& value )
        : value( value )
    {
    }

    const Stats& value;
};

#define DUMP_JSON_MEMBER( name_ ) str << '"' << #name_ << R"json(":)json" << json.value.name_
#define DUMP_JSON_OBJECT( name_ ) str << '"' << #name_ << R"json(":)json" << Json( json.value.name_ )

inline std::ostream& operator<<( std::ostream& str, const Json<GeometryCacheStatistics>& json )
{
    str << '{';
    DUMP_JSON_MEMBER( numTraversables ) << ',';
    DUMP_JSON_MEMBER( numTriangles ) << ',';
    DUMP_JSON_MEMBER( numSpheres ) << ',';
    DUMP_JSON_MEMBER( numNormals ) << ',';
    DUMP_JSON_MEMBER( numUVs ) << ',';
    DUMP_JSON_MEMBER( totalBytesRead ) << ',';
    DUMP_JSON_MEMBER( totalReadTime );
    str << '}';
    return str;
}

inline std::ostream& operator<<( std::ostream& str, const Json<imageSource::CacheStatistics>& json )
{
    str << '{';
    DUMP_JSON_MEMBER( numImageSources ) << ',';
    DUMP_JSON_MEMBER( totalTilesRead ) << ',';
    DUMP_JSON_MEMBER( totalBytesRead ) << ',';
    DUMP_JSON_MEMBER( totalReadTime );
    str << '}';
    return str;
}

inline std::ostream& operator<<( std::ostream& str, const Json<ImageSourceFactoryStatistics>& json )
{
    str << '{';
    DUMP_JSON_OBJECT( fileSources ) << ',';
    DUMP_JSON_OBJECT( alphaSources ) << ',';
    DUMP_JSON_OBJECT( diffuseSources ) << ',';
    DUMP_JSON_OBJECT( skyboxSources );
    str << '}';
    return str;
}

inline std::ostream& operator<<( std::ostream& str, const Json<ProxyFactoryStatistics>& json )
{
    str << '{';
    DUMP_JSON_MEMBER( numSceneProxiesCreated ) << ',';
    DUMP_JSON_MEMBER( numShapeProxiesCreated ) << ',';
    DUMP_JSON_MEMBER( numInstanceProxiesCreated ) << ',';
    DUMP_JSON_MEMBER( numInstanceShapeProxiesCreated ) << ',';
    DUMP_JSON_MEMBER( numInstancePrimitiveProxiesCreated ) << ',';
    DUMP_JSON_MEMBER( numGeometryProxiesCreated );
    str << '}';
    return str;
}

inline std::ostream& operator<<( std::ostream& str, const Json<GeometryResolverStatistics>& json )
{
    str << '{';
    DUMP_JSON_MEMBER( numProxyGeometriesResolved ) << ',';
    DUMP_JSON_MEMBER( numGeometriesRealized );
    str << '}';
    return str;
}

inline std::ostream& operator<<( std::ostream& str, const Json<MaterialResolverStats>& json )
{
    str << '{';
    DUMP_JSON_MEMBER( numPartialMaterialsRealized ) << ',';
    DUMP_JSON_MEMBER( numMaterialsRealized ) << ',';
    DUMP_JSON_MEMBER( numMaterialsReused ) << ',';
    DUMP_JSON_MEMBER( numProxyMaterialsCreated );
    str << '}';
    return str;
}

inline std::ostream& operator<<( std::ostream& str, const Json<std::string>& json )
{
    std::string quoted{ json.value };
    for( auto pos = quoted.find( '\\' ); pos != std::string::npos; pos = quoted.find( '\\', pos + 2 ) )
    {
        quoted.insert( pos, "\\" );
    }
    for( auto pos = quoted.find( '"' ); pos != std::string::npos; pos = quoted.find( '"', pos + 2 ) )
    {
        quoted.insert( pos, "\\" );
    }
    return str << '"' << quoted << '"';
}

inline std::ostream& operator<<( std::ostream& str, const Json<SceneStatistics>& json )
{
    str << '{';
    DUMP_JSON_OBJECT( fileName ) << ',';
    DUMP_JSON_MEMBER( parseTime ) << ',';
    DUMP_JSON_MEMBER( numFreeShapes ) << ',';
    DUMP_JSON_MEMBER( numObjects ) << ',';
    DUMP_JSON_MEMBER( numObjectShapes ) << ',';
    DUMP_JSON_MEMBER( numObjectInstances );
    str << '}';
    return str;
}

inline std::ostream& operator<<( std::ostream& str, const Json<UserInterfaceStatistics>& json )
{
    str << '{';
    DUMP_JSON_MEMBER( numFramesRendered ) << ',';
    DUMP_JSON_OBJECT( geometryCache ) << ',';
    DUMP_JSON_OBJECT( imageSourceFactory ) << ',';
    DUMP_JSON_OBJECT( proxyFactory ) << ',';
    DUMP_JSON_OBJECT( geometry ) << ',';
    DUMP_JSON_OBJECT( materials ) << ',';
    DUMP_JSON_OBJECT( scene );
    str << '}';
    return str;
}

inline std::ostream& operator<<( std::ostream& str, const Json<float3>& json )
{
    return str << '[' << json.value.x << ',' << json.value.y << ',' << json.value.z << ']';
}

inline std::ostream& operator<<( std::ostream& str, const Json<int2>& json )
{
    return str << '[' << json.value.x << ',' << json.value.y << ']';
}

inline std::ostream& operator<<( std::ostream& str, const Json<RenderMode>& json )
{
    switch( json.value )
    {
        case RenderMode::PRIMARY_RAY:
            return str << R"json("primary ray")json";

        case RenderMode::NEAR_AO:
            return str << R"json("near ambient occlusion")json";

        case RenderMode::DISTANT_AO:
            return str << R"json("distant ambient occlusion")json";

        case RenderMode::PATH_TRACING:
            return str << R"json("path tracing")json";
    }
    return str << R"json("?unknown ()json" << std::to_string( static_cast<int>( json.value ) ) << R"json()")json";
}

inline std::ostream& operator<<( std::ostream& str, const Json<ProxyGranularity>& json )
{
    switch( json.value )
    {
        case ProxyGranularity::NONE:
            return str << R"json("none")json";

        case ProxyGranularity::FINE:
            return str << R"json("fine")json";

        case ProxyGranularity::COARSE:
            return str << R"json("coarse")json";
    }
    return str << R"json("?unknown ()json" << std::to_string( static_cast<int>( json.value ) ) << R"json()")json";
}

inline std::ostream& operator<<( std::ostream& str, const Json<bool>& json )
{
    return str << ( json.value ? "true" : "false" );
}

inline std::ostream& operator<<( std::ostream& str, const Json<Options>& json )
{
    str << '{';
    DUMP_JSON_OBJECT( program ) << ',';
    DUMP_JSON_OBJECT( sceneFile ) << ',';
    DUMP_JSON_OBJECT( outFile ) << ',';
    DUMP_JSON_MEMBER( width ) << ',';
    DUMP_JSON_MEMBER( height ) << ',';
    DUMP_JSON_OBJECT( background ) << ',';
    DUMP_JSON_MEMBER( warmupFrames ) << ',';
    DUMP_JSON_OBJECT( oneShotGeometry ) << ',';
    DUMP_JSON_OBJECT( oneShotMaterial ) << ',';
    DUMP_JSON_OBJECT( verboseLoading ) << ',';
    DUMP_JSON_OBJECT( verboseProxyGeometryResolution ) << ',';
    DUMP_JSON_OBJECT( verboseProxyMaterialResolution ) << ',';
    DUMP_JSON_OBJECT( verboseSceneDecomposition ) << ',';
    DUMP_JSON_OBJECT( verboseTextureCreation ) << ',';
    DUMP_JSON_OBJECT( sortProxies ) << ',';
    DUMP_JSON_OBJECT( sync ) << ',';
    DUMP_JSON_OBJECT( usePinholeCamera ) << ',';
    DUMP_JSON_OBJECT( faceForward ) << ',';
    DUMP_JSON_OBJECT( debug ) << ',';
    DUMP_JSON_OBJECT( oneShotDebug ) << ',';
    DUMP_JSON_OBJECT( debugPixel ) << ',';
    DUMP_JSON_OBJECT( renderMode ) << ',';
    DUMP_JSON_OBJECT( proxyGranularity );
    str << '}';
    return str;
}

#undef DUMP_JSON_MEMBER
#undef DUMP_JSON_OBJECT

}  // namespace demandPbrtScene
