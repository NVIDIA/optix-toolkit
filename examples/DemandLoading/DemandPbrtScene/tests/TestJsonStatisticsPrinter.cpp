#include <DemandPbrtScene/JsonStatisticsPrinter.h>

#include <DemandPbrtScene/Options.h>
#include <DemandPbrtScene/UserInterfaceStatistics.h>

#include <gtest/gtest.h>

#include <sstream>

namespace {

class TestJsonStatisticsPrinter : public testing::Test
{
  protected:
    void SetUp() override;

    std::ostringstream                       m_output;
    demandPbrtScene::UserInterfaceStatistics m_stats{};
    std::string                              m_expectedStatsJson;
};

void TestJsonStatisticsPrinter::SetUp()
{
    m_stats.numFramesRendered = 1234;
    m_stats.geometryCache.numTraversables = 1;
    m_stats.geometryCache.numTriangles= 2;
    m_stats.geometryCache.numSpheres = 3;
    m_stats.geometryCache.numNormals = 4;
    m_stats.geometryCache.numUVs = 5;
    m_stats.geometryCache.totalBytesRead = 6U;
    m_stats.geometryCache.totalReadTime = 7.0;
    m_stats.imageSourceFactory.fileSources.numImageSources = 8;
    m_stats.imageSourceFactory.fileSources.totalTilesRead = 9;
    m_stats.imageSourceFactory.fileSources.totalBytesRead = 10;
    m_stats.imageSourceFactory.fileSources.totalReadTime = 11;
    m_stats.proxyFactory.numSceneProxiesCreated = 12;
    m_stats.proxyFactory.numShapeProxiesCreated = 13;
    m_stats.proxyFactory.numInstanceProxiesCreated = 14;
    m_stats.proxyFactory.numInstanceShapeProxiesCreated = 15;
    m_stats.proxyFactory.numInstancePrimitiveProxiesCreated = 16;
    m_stats.proxyFactory.numGeometryProxiesCreated = 17;
    m_stats.geometry.numProxyGeometriesResolved = 18;
    m_stats.geometry.numGeometriesRealized = 19;
    m_stats.materials.numPartialMaterialsRealized = 20;
    m_stats.materials.numMaterialsRealized = 21;
    m_stats.materials.numMaterialsReused = 22;
    m_stats.materials.numProxyMaterialsCreated = 23;
    m_stats.scene.fileName = R"path(C:\scenes\"scene".pbrt)path";
    m_stats.scene.parseTime = 24;
    m_stats.scene.numFreeShapes = 25;
    m_stats.scene.numObjects = 26;
    m_stats.scene.numObjectShapes = 27;
    m_stats.scene.numObjectInstances = 28;

    // If this is a string literal inside EXPECT_EQ it fails to compile on msvc due to the fileName JSON value escapes.
    m_expectedStatsJson =
        // clang-format off
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
        R"json(})json";
    // clang-format on
}

}  // namespace

TEST_F(TestJsonStatisticsPrinter, printMutableStatistics)
{
    m_output << demandPbrtScene::Json( m_stats );

    EXPECT_EQ( m_expectedStatsJson, m_output.str() );
}

TEST_F(TestJsonStatisticsPrinter, printConstStatistics)
{
    const demandPbrtScene::UserInterfaceStatistics stats{ m_stats };

    m_output << demandPbrtScene::Json( stats );

    EXPECT_EQ( m_expectedStatsJson, m_output.str() );
}

TEST_F( TestJsonStatisticsPrinter, renderModePrimaryRay )
{
    m_output << demandPbrtScene::Json( demandPbrtScene::RenderMode::PRIMARY_RAY );

    EXPECT_EQ( R"json("primary ray")json", m_output.str() );
    //PRIMARY_RAY = 0,
    //    NEAR_AO,
    //    DISTANT_AO,
    //    PATH_TRACING
}

TEST_F( TestJsonStatisticsPrinter, renderModeNearAmbientOcclusion )
{
    m_output << demandPbrtScene::Json( demandPbrtScene::RenderMode::NEAR_AO );

    EXPECT_EQ( R"json("near ambient occlusion")json", m_output.str() );
}

TEST_F( TestJsonStatisticsPrinter, renderModeDistantAmbientOcclusion )
{
    m_output << demandPbrtScene::Json( demandPbrtScene::RenderMode::DISTANT_AO );

    EXPECT_EQ( R"json("distant ambient occlusion")json", m_output.str() );
}

TEST_F( TestJsonStatisticsPrinter, renderModePathTracing )
{
    m_output << demandPbrtScene::Json( demandPbrtScene::RenderMode::PATH_TRACING );

    EXPECT_EQ( R"json("path tracing")json", m_output.str() );
}

TEST_F(TestJsonStatisticsPrinter, printOptions)
{
    demandPbrtScene::Options options{};
    options.program = "DemandPbrtScene";
    options.sceneFile = "scene.pbrt";
    options.outFile = "out.png";
    options.background = make_float3(1.0, 2.0, 3.0);
    options.warmupFrames = 4;
    options.oneShotGeometry = true;
    options.oneShotMaterial = true;
    options.verboseLoading = true;
    options.verboseProxyGeometryResolution = true;
    options.verboseProxyMaterialResolution = true;
    options.verboseSceneDecomposition = true;
    options.verboseTextureCreation = true;
    options.sortProxies = true;
    options.sync = true;
    options.usePinholeCamera = true;
    options.faceForward = true;
    options.debug = true;
    options.oneShotDebug = true;
    options.debugPixel = make_int2(5, 6);
    options.renderMode = demandPbrtScene::RenderMode::PATH_TRACING;
    options.proxyGranularity = demandPbrtScene::ProxyGranularity::COARSE;

    m_output << demandPbrtScene::Json(options);

    // clang-format off
    EXPECT_EQ(R"json({)json"
        R"json("program":"DemandPbrtScene",)json"
        R"json("sceneFile":"scene.pbrt",)json"
        R"json("outFile":"out.png",)json"
        R"json("width":768,)json"
        R"json("height":512,)json"
        R"json("background":[1,2,3],)json"
        R"json("warmupFrames":4,)json"
        R"json("oneShotGeometry":true,)json"
        R"json("oneShotMaterial":true,)json"
        R"json("verboseLoading":true,)json"
        R"json("verboseProxyGeometryResolution":true,)json"
        R"json("verboseProxyMaterialResolution":true,)json"
        R"json("verboseSceneDecomposition":true,)json"
        R"json("verboseTextureCreation":true,)json"
        R"json("sortProxies":true,)json"
        R"json("sync":true,)json"
        R"json("usePinholeCamera":true,)json"
        R"json("faceForward":true,)json"
        R"json("debug":true,)json"
        R"json("oneShotDebug":true,)json"
        R"json("debugPixel":[5,6],)json"
        R"json("renderMode":"path tracing",)json"
        R"json("proxyGranularity":"coarse")json"
        R"json(})json", m_output.str());
    // clang-format on
}
