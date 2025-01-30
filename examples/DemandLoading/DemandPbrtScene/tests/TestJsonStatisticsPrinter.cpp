#include <DemandPbrtScene/JsonStatisticsPrinter.h>

#include <DemandPbrtScene/UserInterfaceStatistics.h>

#include <gtest/gtest.h>

#include <sstream>

TEST(TestJsonStatisticsPrinter, printStatistics)
{
    std::ostringstream output;
    demandPbrtScene::UserInterfaceStatistics stats{};
    stats.numFramesRendered = 1234;
    stats.geometryCache.numTraversables = 1;
    stats.geometryCache.numTriangles= 2;
    stats.geometryCache.numSpheres = 3;
    stats.geometryCache.numNormals = 4;
    stats.geometryCache.numUVs = 5;
    stats.geometryCache.totalBytesRead = 6U;
    stats.geometryCache.totalReadTime = 7.0;
    stats.imageSourceFactory.fileSources.numImageSources = 8;
    stats.imageSourceFactory.fileSources.totalTilesRead = 9;
    stats.imageSourceFactory.fileSources.totalBytesRead = 10;
    stats.imageSourceFactory.fileSources.totalReadTime = 11;
    stats.proxyFactory.numSceneProxiesCreated = 12;
    stats.proxyFactory.numShapeProxiesCreated = 13;
    stats.proxyFactory.numInstanceProxiesCreated = 14;
    stats.proxyFactory.numInstanceShapeProxiesCreated = 15;
    stats.proxyFactory.numInstancePrimitiveProxiesCreated = 16;
    stats.proxyFactory.numGeometryProxiesCreated = 17;
    stats.geometry.numProxyGeometriesResolved = 18;
    stats.geometry.numGeometriesRealized = 19;
    stats.materials.numPartialMaterialsRealized = 20;
    stats.materials.numMaterialsRealized = 21;
    stats.materials.numMaterialsReused = 22;
    stats.materials.numProxyMaterialsCreated = 23;
    stats.scene.fileName = R"path(C:\scenes\"scene".pbrt)path";
    stats.scene.parseTime = 24;
    stats.scene.numFreeShapes = 25;
    stats.scene.numObjects = 26;
    stats.scene.numObjectShapes = 27;
    stats.scene.numObjectInstances = 28;

    output << demandPbrtScene::Json( stats );

    // clang-format off
    // If this is a string literal inside EXPECT_EQ it fails to compile on msvc due to the fileName JSON value escapes.
    constexpr const char*expected{
        R"json({)json"
        R"json("numFramesRendered":1234,)json"
        R"json("geometryCache":{"numTraversables":1,"numTriangles":2,"numSpheres":3,"numNormals":4,"numUVs":5,"totalBytesRead":6,"totalReadTime":7},)json"
        R"json("imageSourceFactory":{)json"
        R"json("fileSources":{"numImageSources":8,"totalTilesRead":9,"totalBytesRead":10,"totalReadTime":11},)json"
        R"json("alphaSources":{"numImageSources":0,"totalTilesRead":0,"totalBytesRead":0,"totalReadTime":0},)json"
        R"json("diffuseSources":{"numImageSources":0,"totalTilesRead":0,"totalBytesRead":0,"totalReadTime":0},)json"
        R"json("skyboxSources":{"numImageSources":0,"totalTilesRead":0,"totalBytesRead":0,"totalReadTime":0})json"
        R"json(},)json"
        R"json("proxyFactory":{)json"
        R"json("numSceneProxiesCreated":12,)json"
        R"json("numShapeProxiesCreated":13,)json"
        R"json("numInstanceProxiesCreated":14,)json"
        R"json("numInstanceShapeProxiesCreated":15,)json"
        R"json("numInstancePrimitiveProxiesCreated":16,)json"
        R"json("numGeometryProxiesCreated":17)json"
        R"json(},)json"
        R"json("geometry":{)json"
        R"json("numProxyGeometriesResolved":18,)json"
        R"json("numGeometriesRealized":19)json"
        R"json(},)json"
        R"json("materials":{)json"
        R"json("numPartialMaterialsRealized":20,)json"
        R"json("numMaterialsRealized":21,)json"
        R"json("numMaterialsReused":22,)json"
        R"json("numProxyMaterialsCreated":23)json"
        R"json(},)json"
        R"json("scene":{)json"
        R"json("fileName":"C:\\scenes\\\"scene\".pbrt",)json"
        R"json("parseTime":24,)json"
        R"json("numFreeShapes":25,)json"
        R"json("numObjects":26,)json"
        R"json("numObjectShapes":27,)json"
        R"json("numObjectInstances":28)json"
        R"json(})json"
        R"json(})json"};
    // clang-format on
    EXPECT_EQ( expected, output.str() );
}
