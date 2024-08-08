//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

// gtest has to come before pbrt stuff
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Matchers.h"
#include "MockDemandTextureCache.h"
#include "MockGeometryLoader.h"
#include "ParamsPrinters.h"
#include "SceneAdapters.h"

#include <DemandTextureCache.h>
#include <GeometryResolver.h>
#include <MaterialResolver.h>
#include <Options.h>
#include <Params.h>
#include <PbrtScene.h>
#include <ProgramGroups.h>
#include <Renderer.h>

#include <OptiXToolkit/DemandGeometry/Mocks/Matchers.h>
#include <OptiXToolkit/DemandGeometry/Mocks/MockDemandLoader.h>
#include <OptiXToolkit/DemandGeometry/Mocks/MockOptix.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneLoader.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <iterator>

using namespace testing;
using namespace otk::testing;
using namespace demandPbrtScene;
using namespace demandPbrtScene::testing;

using P3 = pbrt::Point3f;
using V3 = pbrt::Vector3f;
using B3 = pbrt::Bounds3f;

using Stats = SceneStatistics;

constexpr const char* ALPHA_MAP_FILENAME{ "alphaMap.png" };
constexpr const char* DIFFUSE_MAP_FILENAME{ "diffuseMap.png" };

inline float3 fromPoint3f( const ::P3& pt )
{
    return make_float3( pt.x, pt.y, pt.z );
};
inline float3 fromVector3f( const ::V3& vec )
{
    return make_float3( vec.x, vec.y, vec.z );
};

#define OUTPUT_ENUM( enum_ )                                                                                           \
    case enum_:                                                                                                        \
        return str << #enum_ << " (" << static_cast<int>( enum_ ) << ')'

inline std::ostream& operator<<( std::ostream& str, CUaddress_mode val )
{
    switch( val )
    {
        OUTPUT_ENUM( CU_TR_ADDRESS_MODE_WRAP );
        OUTPUT_ENUM( CU_TR_ADDRESS_MODE_CLAMP );
        OUTPUT_ENUM( CU_TR_ADDRESS_MODE_MIRROR );
        OUTPUT_ENUM( CU_TR_ADDRESS_MODE_BORDER );
    }
    return str << "? (" << static_cast<int>( val ) << ')';
}

inline std::ostream& operator<<( std::ostream& str, CUfilter_mode val )
{
    switch( val )
    {
        OUTPUT_ENUM( CU_TR_FILTER_MODE_POINT );
        OUTPUT_ENUM( CU_TR_FILTER_MODE_LINEAR );
    }
    return str << "? (" << static_cast<int>( val ) << ')';
}

namespace demandLoading {

inline std::ostream& operator<<( std::ostream& str, const TextureDescriptor& val )
{
    const char fill = str.fill();
    return str << "TextureDescriptor{ " << val.addressMode[0] << ", " << val.addressMode[1] << ", " << val.filterMode
               << ", " << val.mipmapFilterMode << ", " << val.maxAnisotropy << ", " << val.flags << " (0x" << std::hex
               << std::setw( 2 * sizeof( val.flags ) ) << std::setfill( '0' ) << val.flags << std::dec << std::setw( 0 )
               << std::setfill( fill ) << ") }";
}

template <typename T>
std::ostream& operator<<( std::ostream& str, const DeviceArray<T>& val )
{
    return str << "DeviceArray{" << static_cast<void*>( val.data ) << ", " << val.capacity << "}";
}

template <typename T>
bool operator==( const DeviceArray<T>& lhs, const DeviceArray<T>& rhs )
{
    return lhs.data == rhs.data && lhs.capacity == rhs.capacity;
}

template <typename T>
bool operator!=( const DeviceArray<T>& lhs, const DeviceArray<T>& rhs )
{
    return !( lhs == rhs );
}

std::ostream& operator<<( std::ostream& str, const DeviceContext& val )
{
    return str << "DeviceContext{ "              //
               << val.pageTable                  //
               << ", " << val.maxNumPages        //
               << ", " << val.referenceBits      //
               << ", " << val.residenceBits      //
               << ", " << val.lruTable           //
               << ", " << val.requestedPages     //
               << ", " << val.stalePages         //
               << ", " << val.evictablePages     //
               << ", " << val.arrayLengths       //
               << ", " << val.filledPages        //
               << ", " << val.invalidatedPages   //
               << ", " << val.requestIfResident  //
               << ", " << val.poolIndex          //
               << "}";                           //
}

inline bool operator==( const DeviceContext& lhs, const DeviceContext& rhs )
{
    // clang-format off
    return lhs.pageTable         == rhs.pageTable
        && lhs.maxNumPages       == rhs.maxNumPages
        && lhs.referenceBits     == rhs.referenceBits
        && lhs.residenceBits     == rhs.residenceBits
        && lhs.lruTable          == rhs.lruTable
        && lhs.requestedPages    == rhs.requestedPages
        && lhs.stalePages        == rhs.stalePages
        && lhs.evictablePages    == rhs.evictablePages
        && lhs.arrayLengths      == rhs.arrayLengths
        && lhs.filledPages       == rhs.filledPages
        && lhs.invalidatedPages  == rhs.invalidatedPages
        && lhs.requestIfResident == rhs.requestIfResident 
        && lhs.poolIndex         == rhs.poolIndex;
    // clang-format on
}

inline bool operator!=( const DeviceContext& lhs, const DeviceContext& rhs )
{
    return !( lhs == rhs );
}

}  // namespace demandLoading

namespace otk {

std::ostream& operator<<( std::ostream& str, const Transform4& transform )
{
    str << "[ ";
    for( int row = 0; row < 4; ++row )
    {
        str << "[ " << transform.m[row] << " ]";
        if( row != 3 )
        {
            str << ", ";
        }
    }
    return str << " ]";
}

}  // namespace otk

static demandGeometry::Context fakeDemandGeometryContext()
{
    return demandGeometry::Context{ reinterpret_cast<OptixAabb*>( static_cast<std::uintptr_t>( 0xdeadbeefU ) ) };
}

static demandLoading::DeviceContext fakeDemandLoadingDeviceContext()
{
    demandLoading::DeviceContext demandContext{};
    demandContext.residenceBits = reinterpret_cast<unsigned int*>( 0xf00f00ULL );
    return demandContext;
}

static OptixDeviceContext fakeOptixDeviceContext()
{
    return reinterpret_cast<OptixDeviceContext>( static_cast<std::intptr_t>( 0xfeedfeed ) );
}

inline B3 toBounds3( const OptixAabb& bounds )
{
    return B3{ P3{ bounds.minX, bounds.minY, bounds.minZ }, P3{ bounds.maxX, bounds.maxY, bounds.maxZ } };
}

inline OptixProgramGroup PG( unsigned int id )
{
    return reinterpret_cast<OptixProgramGroup>( static_cast<std::intptr_t>( id ) );
};

namespace {

class MockSceneLoader : public StrictMock<otk::pbrt::SceneLoader>
{
  public:
    ~MockSceneLoader() override = default;

    MOCK_METHOD( SceneDescriptionPtr, parseFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( SceneDescriptionPtr, parseString, ( const std::string& str ), ( override ) );
};

class MockGeometryResolver : public StrictMock<GeometryResolver>
{
  public:
    ~MockGeometryResolver() override = default;

    MOCK_METHOD( GeometryResolverStatistics, getStatistics, (), ( const, override ) );
    MOCK_METHOD( void, initialize, (CUstream, OptixDeviceContext, const SceneDescriptionPtr&, SceneSyncState&), ( override ) );
    MOCK_METHOD( demandGeometry::Context, getContext, (), ( const, override ) );
    MOCK_METHOD( void, resolveOneGeometry, (), ( override ) );
    MOCK_METHOD( bool, resolveRequestedProxyGeometries, (CUstream, OptixDeviceContext, const FrameStopwatch&, SceneSyncState&), ( override ) );
};

class MockMaterialResolver : public StrictMock<MaterialResolver>
{
  public:
    ~MockMaterialResolver() override = default;

    MOCK_METHOD( MaterialResolverStats, getStatistics, (), ( const, override ) );
    MOCK_METHOD( bool, resolveMaterialForGeometry, (uint_t, SceneGeometry&, SceneSyncState&), ( override ) );
    MOCK_METHOD( void, resolveOneMaterial, (), ( override ) );
    MOCK_METHOD( MaterialResolution, resolveRequestedProxyMaterials, (CUstream, const FrameStopwatch&, SceneSyncState&), ( override ) );
};

class MockProgramGroups : public StrictMock<ProgramGroups>
{
  public:
    ~MockProgramGroups() override = default;

    MOCK_METHOD( void, cleanup, (), ( override ) );
    MOCK_METHOD( uint_t, getRealizedMaterialSbtOffset, (const GeometryInstance&), ( override ) );
    MOCK_METHOD( void, initialize, (), ( override ) );
};

class MockRenderer : public StrictMock<Renderer>
{
  public:
    ~MockRenderer() override = default;

    MOCK_METHOD( void, initialize, ( CUstream ), ( override ) );
    MOCK_METHOD( void, cleanup, (), ( override ) );
    MOCK_METHOD( const otk::DebugLocation&, getDebugLocation, (), ( const override ) );
    MOCK_METHOD( const LookAtParams&, getLookAt, (), ( const override ) );
    MOCK_METHOD( const PerspectiveCamera&, getCamera, (), ( const override ) );
    MOCK_METHOD( Params&, getParams, (), ( override ) );
    MOCK_METHOD( OptixDeviceContext, getDeviceContext, (), ( const, override ) );
    MOCK_METHOD( const OptixPipelineCompileOptions&, getPipelineCompileOptions, (), ( const, override ) );
    MOCK_METHOD( void, setDebugLocation, (const otk::DebugLocation&), ( override ) );
    MOCK_METHOD( void, setCamera, (const PerspectiveCamera&), ( override ) );
    MOCK_METHOD( void, setLookAt, (const LookAtParams&), ( override ) );
    MOCK_METHOD( void, setProgramGroups, (const std::vector<OptixProgramGroup>&), ( override ) );
    MOCK_METHOD( void, beforeLaunch, ( CUstream ), ( override ) );
    MOCK_METHOD( void, launch, (CUstream, uchar4*), ( override ) );
    MOCK_METHOD( void, afterLaunch, (), ( override ) );
    MOCK_METHOD( void, fireOneDebugDump, (), ( override ) );
    MOCK_METHOD( void, setClearAccumulator, (), ( override ) );
};

using StrictMockDemandLoader    = StrictMock<MockDemandLoader>;
using StrictMockOptix           = StrictMock<MockOptix>;
using MockSceneLoaderPtr        = std::shared_ptr<MockSceneLoader>;
using MockDemandLoaderPtr       = std::shared_ptr<StrictMockDemandLoader>;
using MockGeometryResolverPtr   = std::shared_ptr<MockGeometryResolver>;
using MockMaterialResolverPtr   = std::shared_ptr<MockMaterialResolver>;
using MockProgramGroupsPtr      = std::shared_ptr<MockProgramGroups>;
using MockRendererPtr           = std::shared_ptr<MockRenderer>;

using AccelBuildOptionsMatcher = Matcher<const OptixAccelBuildOptions*>;
using BuildInputMatcher        = Matcher<const OptixBuildInput*>;

MATCHER_P( hasEye, value, "" )
{
    const float3 expected{ fromPoint3f( value ) };
    if( arg.eye != expected )
    {
        *result_listener << "expected eye point " << expected << ", got " << arg.eye;
        return false;
    }

    *result_listener << "has eye point " << expected;
    return true;
}

MATCHER_P( hasLookAt, value, "" )
{
    const float3 expected{ fromPoint3f( value ) };
    if( arg.lookAt != expected )
    {
        *result_listener << "expected look at point " << expected << ", got " << arg.lookAt;
        return false;
    }

    *result_listener << "has look at point " << expected;
    return true;
}

MATCHER_P( hasUp, value, "" )
{
    const float3 expected{ fromVector3f( value ) };
    if( arg.up != expected )
    {
        *result_listener << "expected up vector " << expected << ", got " << arg.up;
        return false;
    }

    *result_listener << "has up vector " << expected;
    return true;
}

MATCHER_P( hasCameraToWorldTransform, pbrtTransform, "" )
{
    otk::Transform4 lhs;
    toFloat4Transform( lhs.m, pbrtTransform );
    const otk::Transform4 rhs{ arg.cameraToWorld };
    if( lhs != rhs )
    {
        *result_listener << "expected camera to world transform " << lhs << ", got " << rhs;
        return false;
    }

    *result_listener << "has camera to world transform " << lhs;
    return true;
}

MATCHER_P( hasWorldToCameraTransform, pbrtTransform, "" )
{
    otk::Transform4 lhs;
    toFloat4Transform( lhs.m, pbrtTransform );
    const otk::Transform4 rhs{ arg.worldToCamera };
    if( lhs != rhs )
    {
        *result_listener << "expected world to camera transform " << lhs << ", got " << rhs;
        return false;
    }

    *result_listener << "has world to camera transform " << lhs;
    return true;
}

MATCHER_P( hasCameraToScreenTransform, pbrtTransform, "" )
{
    otk::Transform4 lhs;
    toFloat4Transform( lhs.m, pbrtTransform );
    const otk::Transform4 rhs{ arg.cameraToScreen };
    if( lhs != rhs )
    {
        *result_listener << "expected camera to screen transform " << lhs << ", got " << rhs;
        return false;
    }

    *result_listener << "has camera to screen transform " << lhs;
    return true;
}

MATCHER_P( hasFov, fov, "" )
{
    if( arg.fovY != fov )
    {
        *result_listener << "expected field of view angle " << fov << ", got " << arg.fovY;
        return false;
    }

    *result_listener << "has field of view angle " << fov;
    return true;
}

MATCHER_P( hasFocalDistance, value, "" )
{
    if( arg.focalDistance != value )
    {
        *result_listener << "expected focal distance " << value << ", got " << arg.focalDistance;
        return false;
    }

    *result_listener << "has focal distance " << value;
    return true;
}

MATCHER_P( hasLensRadius, value, "" )
{
    if( arg.lensRadius != value )
    {
        *result_listener << "expected lens radius " << value << ", got " << arg.lensRadius;
        return false;
    }

    *result_listener << "has lens radius " << value;
    return true;
}

bool hasDirectionalLight( MatchResultListener* result_listener, const DirectionalLight& light, const DirectionalLight& actual )
{
    if( actual != light )
    {
        *result_listener << "expected directional light " << light << ", got " << actual;
        return false;
    }
    *result_listener << "has expected directional light " << light;
    return true;
}

MATCHER_P( hasDeviceDirectionalLight, light, "" )
{
    if( arg == nullptr )
    {
        *result_listener << "expected non-null argument";
        return false;
    }

    DirectionalLight actual{};
    OTK_ERROR_CHECK( cudaMemcpy( &actual, arg, sizeof( DirectionalLight ), cudaMemcpyDeviceToHost ) );
    return hasDirectionalLight( result_listener, light, actual );
}

bool hasInfiniteLight( MatchResultListener* result_listener, const InfiniteLight& light, const InfiniteLight& actual )
{
    if( actual != light )
    {
        *result_listener << "expected infinite light " << light << ", got " << actual;
        return false;
    }
    *result_listener << "has expected infinite light " << light;
    return true;
}

MATCHER_P( hasDeviceInfiniteLight, light, "" )
{
    if( arg == nullptr )
    {
        *result_listener << "expected non-null argument";
        return false;
    }

    InfiniteLight actual{};
    OTK_ERROR_CHECK( cudaMemcpy( &actual, arg, sizeof( InfiniteLight ), cudaMemcpyDeviceToHost ) );
    return hasInfiniteLight( result_listener, light, actual );
}

// This was needed to satisfy gcc instead of constructing from a brace initializer list.
Options testOptions()
{
    Options options{};
    options.sceneFile = "test.pbrt";
    return options;
}

class TestPbrtScene : public Test
{
  public:
    ~TestPbrtScene() override = default;

  protected:
    void SetUp() override;
    void TearDown() override;

    Expectation expectAccelComputeMemoryUsage( const AccelBuildOptionsMatcher& options, const BuildInputMatcher& buildInput );
    Expectation expectAccelBuild( const AccelBuildOptionsMatcher& options, const BuildInputMatcher& buildInput, OptixTraversableHandle result );
    ExpectationSet expectInitializeCreatesOptixState();
    ExpectationSet expectInitialize();

    CUstream                  m_stream{};
    StrictMockOptix           m_optix{};
    MockSceneLoaderPtr        m_sceneLoader{ std::make_shared<MockSceneLoader>() };
    MockDemandTextureCachePtr m_demandTextureCache{ createMockDemandTextureCache() };
    MockDemandLoaderPtr       m_demandLoader{ std::make_shared<StrictMockDemandLoader>() };
    MockProgramGroupsPtr      m_programGroups{ std::make_shared<MockProgramGroups>() };
    MockMaterialResolverPtr   m_materialResolver{ std::make_shared<MockMaterialResolver>() };
    MockGeometryResolverPtr   m_geometryResolver{ std::make_shared<MockGeometryResolver>() };
    MockRendererPtr           m_renderer{ std::make_shared<MockRenderer>() };
    Options                   m_options{ testOptions() };
    // clang-format off
    PbrtScene m_scene{ m_options, m_sceneLoader, m_demandTextureCache, m_demandLoader, m_programGroups, m_materialResolver, m_geometryResolver, m_renderer };
    // clang-format on
    SceneDescriptionPtr          m_sceneDesc{ std::make_shared<otk::pbrt::SceneDescription>() };
    OptixAabb                    m_sceneBounds{ -1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f };
    demandGeometry::Context      m_demandGeomContext{ fakeDemandGeometryContext() };
    demandLoading::DeviceContext m_demandLoadContext{ fakeDemandLoadingDeviceContext() };
    OptixTraversableHandle       m_fakeProxyTraversable{ 0xbaddf00dU };
    OptixTraversableHandle       m_fakeTopLevelTraversable{ 0xf01df01dU };
    OptixDeviceContext           m_fakeContext{ fakeOptixDeviceContext() };
    OptixAccelBufferSizes        m_topLevelASSizes{};
    OptixPipelineCompileOptions  m_pipelineCompileOptions{};
};

void TestPbrtScene::SetUp()
{
    Test::SetUp();
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    OTK_ERROR_CHECK( cuStreamCreate( &m_stream, 0 ) );
    initMockOptix( m_optix );

    m_options.sceneFile = "cube.pbrt";

    otk::pbrt::LookAtDefinition&            lookAt = m_sceneDesc->lookAt;
    otk::pbrt::PerspectiveCameraDefinition& camera = m_sceneDesc->camera;

    lookAt.lookAt         = P3( 111.0f, 222.0f, 3333.0f );
    lookAt.eye            = P3( 444.0f, 555.0f, 666.0f );
    lookAt.up             = Normalize( V3( 1.0f, 2.0f, 3.0f ) );
    camera.fov            = 45.0f;
    camera.focalDistance  = 3000.0f;
    camera.lensRadius     = 0.125f;
    camera.cameraToWorld  = LookAt( lookAt.eye, lookAt.lookAt, lookAt.up );
    camera.cameraToScreen = pbrt::Perspective( camera.fov, 1e-2f, 1000.f );
    m_sceneDesc->bounds   = toBounds3( m_sceneBounds );

    m_topLevelASSizes.tempSizeInBytes   = 1234U;
    m_topLevelASSizes.outputSizeInBytes = 5678U;

    m_pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
}

void TestPbrtScene::TearDown()
{
    OTK_ERROR_CHECK( cuStreamDestroy( m_stream ) );
}

Expectation TestPbrtScene::expectAccelComputeMemoryUsage( const AccelBuildOptionsMatcher& options, const BuildInputMatcher& buildInput )
{
    return EXPECT_CALL( m_optix, accelComputeMemoryUsage( m_fakeContext, options, buildInput, _, _ ) )
        .WillOnce( DoAll( SetArgPointee<4>( m_topLevelASSizes ), Return( OPTIX_SUCCESS ) ) );
}

Expectation TestPbrtScene::expectAccelBuild( const AccelBuildOptionsMatcher& options, const BuildInputMatcher& buildInput, OptixTraversableHandle result )
{
    return EXPECT_CALL( m_optix, accelBuild( m_fakeContext, m_stream, options, buildInput, _, _, _, _, _, _, nullptr, 0 ) )
        .WillOnce( DoAll( SetArgPointee<9>( result ), Return( OPTIX_SUCCESS ) ) );
}

ExpectationSet TestPbrtScene::expectInitializeCreatesOptixState()
{
    ExpectationSet expect;
    expect += EXPECT_CALL( *m_sceneLoader, parseFile( _ ) ).WillOnce( Return( m_sceneDesc ) );
    expect += EXPECT_CALL( *m_programGroups, initialize() );
    // This getter can be called anytime in any order.
    EXPECT_CALL( *m_renderer, getDeviceContext() ).WillRepeatedly( Return( m_fakeContext ) );
    expect += EXPECT_CALL( *m_geometryResolver, initialize( m_stream, _, m_sceneDesc, _ ) )
                  .WillOnce( [=]( CUstream, OptixDeviceContext, const SceneDescriptionPtr&, SceneSyncState& sync ) {
                      const uint_t  fakeInstanceId{ 1964 };
                      OptixInstance instance{};
                      instance.traversableHandle = m_fakeProxyTraversable;
                      instance.sbtOffset         = +HitGroupIndex::PROXY_GEOMETRY;
                      instance.instanceId        = fakeInstanceId;
                      sync.topLevelInstances.push_back( instance );
                  } );
    expect += expectAccelComputeMemoryUsage( _, _ );
    expect += expectAccelBuild( _, _, m_fakeTopLevelTraversable );
    return expect;
}

ExpectationSet TestPbrtScene::expectInitialize()
{
    ExpectationSet expect = expectInitializeCreatesOptixState();
    expect += EXPECT_CALL( *m_renderer, setLookAt( _ ) );
    expect += EXPECT_CALL( *m_renderer, setCamera( _ ) );
    return expect;
}

class TestPbrtSceneInitialized : public TestPbrtScene
{
  public:
    ~TestPbrtSceneInitialized() override = default;

  protected:
    void SetUp() override;

    ExpectationSet expectCreateTopLevelTraversableAfter( const BuildInputMatcher& buildInput,
                                                         OptixTraversableHandle   result,
                                                         const ExpectationSet&    before );
    Expectation    expectProxyMaterialsResolvedAfter( MaterialResolution resolution, const ExpectationSet& before );
    Expectation    expectLaunchPrepareTrueAfter( const ExpectationSet& before );
    Expectation    expectNoMaterialResolvedAfter( const ExpectationSet& before );
    Expectation    expectNoGeometryResolvedAfter( const ExpectationSet& before );
    void           setDistantLightOnSceneDescription();
    void           setInfiniteLightOnSceneDescription();

    ExpectationSet         m_init;
    OptixTraversableHandle m_fakeTriMeshTraversable{ 0xbeefbeefU };
    uint_t                 m_fakeMaterialId{ 666U };
    DirectionalLight       m_expectedDirectionalLight{};
    InfiniteLight          m_expectedInfiniteLight{};
};

void TestPbrtSceneInitialized::SetUp()
{
    TestPbrtScene::SetUp();
    m_init = expectInitialize();
    m_scene.initialize( m_stream );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).WillRepeatedly( Return( m_fakeContext ) );
    EXPECT_CALL( *m_geometryResolver, getContext() ).WillRepeatedly( Return( m_demandGeomContext ) );
}

ExpectationSet TestPbrtSceneInitialized::expectCreateTopLevelTraversableAfter( const BuildInputMatcher& buildInput,
                                                                               OptixTraversableHandle   result,
                                                                               const ExpectationSet&    before )
{
    ExpectationSet set;
    set += EXPECT_CALL( m_optix, accelComputeMemoryUsage( m_fakeContext, NotNull(), buildInput, _, _ ) )
               .After( before )
               .WillOnce( DoAll( SetArgPointee<4>( m_topLevelASSizes ), Return( OPTIX_SUCCESS ) ) );
    set += EXPECT_CALL( m_optix, accelBuild( m_fakeContext, m_stream, NotNull(), buildInput, _, _, _, _, _, _, nullptr, 0 ) )
               .After( before )
               .WillOnce( DoAll( SetArgPointee<9>( result ), Return( OPTIX_SUCCESS ) ) );
    return set;
}

Expectation TestPbrtSceneInitialized::expectProxyMaterialsResolvedAfter( MaterialResolution resolution, const ExpectationSet& before )
{
    return EXPECT_CALL( *m_materialResolver, resolveRequestedProxyMaterials( _, _, _ ) ).After( before ).WillOnce( Return( resolution ) );
}

Expectation TestPbrtSceneInitialized::expectLaunchPrepareTrueAfter( const ExpectationSet& before )
{
    return EXPECT_CALL( *m_demandLoader, launchPrepare( m_stream, _ ) )
        .After( before )
        .WillOnce( DoAll( SetArgReferee<1>( m_demandLoadContext ), Return( true ) ) );
}

Expectation TestPbrtSceneInitialized::expectNoMaterialResolvedAfter( const ExpectationSet& before )
{
    return expectProxyMaterialsResolvedAfter( MaterialResolution::NONE, before );
}

Expectation TestPbrtSceneInitialized::expectNoGeometryResolvedAfter( const ExpectationSet& before )
{
    return EXPECT_CALL( *m_geometryResolver, resolveRequestedProxyGeometries( m_stream, _, _, _ ) ).After( before ).WillOnce( Return( false ) );
}

void TestPbrtSceneInitialized::setDistantLightOnSceneDescription()
{
    const P3              color( 1.0f, 0.2f, 0.4f );
    const P3              scale( 2.0f, 2.0f, 2.0f );
    const V3              direction( 1.0f, 1.0f, 1.0f );
    const pbrt::Transform lightToWorld;
    m_sceneDesc->distantLights.push_back( ::otk::pbrt::DistantLightDefinition{ scale, color, direction, lightToWorld } );
    DirectionalLight& light = m_expectedDirectionalLight;
    light.color             = fromPoint3f( color ) * fromPoint3f( scale );
    light.direction         = fromVector3f( Normalize( lightToWorld( direction ) ) );
}

void TestPbrtSceneInitialized::setInfiniteLightOnSceneDescription()
{
    const P3              color( 1.0f, 0.2f, 0.4f );
    const P3              scale( 2.0f, 2.0f, 2.0f );
    const pbrt::Transform lightToWorld;
    m_sceneDesc->infiniteLights.push_back( ::otk::pbrt::InfiniteLightDefinition{ scale, color, 1, "", lightToWorld } );
    InfiniteLight& light = m_expectedInfiniteLight;
    light.color          = fromPoint3f( color );
    light.scale          = fromPoint3f( scale );
}

}  // namespace

TEST_F( TestPbrtScene, initializeCreatesOptixResourcesForLoadedScene )
{
    EXPECT_CALL( *m_sceneLoader, parseFile( m_options.sceneFile ) ).Times( 1 ).WillOnce( Return( m_sceneDesc ) );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).WillRepeatedly( Return( m_fakeContext ) );
    EXPECT_CALL( *m_programGroups, initialize() );
    const uint_t fakeInstanceId{ 1964 };
    EXPECT_CALL( *m_geometryResolver, initialize( m_stream, _, m_sceneDesc, _ ) )
        .WillOnce( [=]( CUstream, OptixDeviceContext, const SceneDescriptionPtr&, SceneSyncState& sync ) {
            OptixInstance instance{};
            instance.traversableHandle = m_fakeProxyTraversable;
            instance.sbtOffset         = +HitGroupIndex::PROXY_GEOMETRY;
            instance.instanceId        = fakeInstanceId;
            sync.topLevelInstances.push_back( instance );
        } );
    auto isIAS =
        AllOf( NotNull(),
               hasInstanceBuildInput( 0, hasAll( hasNumInstances( 1 ),
                                                 hasDeviceInstances( hasInstance( 0, hasInstanceTraversable( m_fakeProxyTraversable ),
                                                                                  hasInstanceSbtOffset( +HitGroupIndex::PROXY_GEOMETRY ),
                                                                                  hasInstanceId( fakeInstanceId ) ) ) ) ) );
    expectAccelComputeMemoryUsage( NotNull(), isIAS );
    expectAccelBuild( NotNull(), isIAS, m_fakeTopLevelTraversable );
    const otk::pbrt::LookAtDefinition&            lookAt = m_sceneDesc->lookAt;
    const otk::pbrt::PerspectiveCameraDefinition& camera = m_sceneDesc->camera;
    EXPECT_CALL( *m_renderer, setLookAt( AllOf( hasEye( lookAt.eye ), hasLookAt( lookAt.lookAt ), hasUp( lookAt.up ) ) ) );
    EXPECT_CALL( *m_renderer, setCamera( AllOf( hasCameraToWorldTransform( camera.cameraToWorld ),
                                                hasWorldToCameraTransform( Inverse( camera.cameraToWorld ) ),
                                                hasCameraToScreenTransform( camera.cameraToScreen ), hasFov( camera.fov ) ) ) );

    m_scene.initialize( m_stream );
}

TEST_F( TestPbrtScene, initializeSetsDefaultCameraWhenMissingFromScene )
{
    expectInitializeCreatesOptixState();
    const otk::pbrt::LookAtDefinition& lookAt = m_sceneDesc->lookAt;
    EXPECT_CALL( *m_renderer, setLookAt( AllOf( hasEye( lookAt.eye ), hasLookAt( lookAt.lookAt ), hasUp( lookAt.up ) ) ) );
    const otk::pbrt::PerspectiveCameraDefinition& camera = m_sceneDesc->camera;
    EXPECT_CALL( *m_renderer,
                 setCamera( AllOf( hasFov( camera.fov ), hasFocalDistance( camera.focalDistance ),
                                   hasLensRadius( camera.lensRadius ), hasCameraToWorldTransform( camera.cameraToWorld ) ) ) );

    m_scene.initialize( m_stream );
}

TEST_F( TestPbrtSceneInitialized, beforeLaunchSetsInitialParams )
{
    expectLaunchPrepareTrueAfter( m_init );
    expectNoMaterialResolvedAfter( m_init );
    expectNoGeometryResolvedAfter( m_init );

    Params params{};
    m_scene.beforeLaunch( m_stream, params );

    EXPECT_NE( float3{}, params.ambientColor );
    for( int i = 0; i < 6; ++i )
    {
        EXPECT_NE( float3{}, params.proxyFaceColors[i] ) << "proxy face " << i;
    }
    EXPECT_NE( 0.0f, params.sceneEpsilon );
    EXPECT_NE( OptixTraversableHandle{}, params.traversable );
    EXPECT_EQ( m_demandLoadContext, params.demandContext );
    EXPECT_EQ( m_demandGeomContext, params.demandGeomContext );
    EXPECT_NE( float3{}, params.demandMaterialColor );
    // no realized materials yet
    EXPECT_EQ( nullptr, params.realizedMaterials );
}

TEST_F( TestPbrtSceneInitialized, beforeLaunchSetsDirectionalLightsInParams )
{
    setDistantLightOnSceneDescription();
    expectLaunchPrepareTrueAfter( m_init );
    expectNoMaterialResolvedAfter( m_init );
    expectNoGeometryResolvedAfter( m_init );

    Params params{};
    m_scene.beforeLaunch( m_stream, params );

    ASSERT_EQ( 1, params.numDirectionalLights );
    EXPECT_THAT( params.directionalLights, hasDeviceDirectionalLight( m_expectedDirectionalLight ) );
}

TEST_F( TestPbrtSceneInitialized, beforeLaunchSetsInfiniteLightsInParams )
{
    setInfiniteLightOnSceneDescription();
    expectLaunchPrepareTrueAfter( m_init );
    expectNoMaterialResolvedAfter( m_init );
    expectNoGeometryResolvedAfter( m_init );

    Params params{};
    m_scene.beforeLaunch( m_stream, params );

    ASSERT_EQ( 1, params.numInfiniteLights );
    EXPECT_THAT( params.infiniteLights, hasDeviceInfiniteLight( m_expectedInfiniteLight ) );
    for( int i = 0; i < 6; ++i )
    {
        EXPECT_NE( float3{}, params.proxyFaceColors[i] ) << "proxy face " << i;
    }
    EXPECT_NE( 0.0f, params.sceneEpsilon );
    EXPECT_NE( OptixTraversableHandle{}, params.traversable );
    EXPECT_EQ( m_demandLoadContext, params.demandContext );
    EXPECT_EQ( m_demandGeomContext, params.demandGeomContext );
    EXPECT_NE( float3{}, params.demandMaterialColor );
    // no realized materials yet
    EXPECT_EQ( nullptr, params.realizedMaterials );
}

TEST_F( TestPbrtSceneInitialized, beforeLaunchCreatesSkyboxForInfiniteLightsInParams )
{
    setInfiniteLightOnSceneDescription();
    const std::string path{ "foo.exr" };
    m_sceneDesc->infiniteLights[0].environmentMapName = path;
    expectLaunchPrepareTrueAfter( m_init );
    expectNoMaterialResolvedAfter( m_init );
    expectNoGeometryResolvedAfter( m_init );
    const uint_t textureId{ 1234 };
    EXPECT_CALL( *m_demandTextureCache, createSkyboxTextureFromFile( path ) ).WillOnce( Return( textureId ) );

    Params params{};
    m_scene.beforeLaunch( m_stream, params );

    ASSERT_EQ( 1, params.numInfiniteLights );
    m_expectedInfiniteLight.skyboxTextureId = textureId;
    EXPECT_THAT( params.infiniteLights, hasDeviceInfiniteLight( m_expectedInfiniteLight ) );
}

TEST_F( TestPbrtSceneInitialized, afterLaunchProcessesRequests )
{
    demandLoading::Ticket ticket;
    Params                params{};
    EXPECT_CALL( *m_demandLoader, processRequests( m_stream, params.demandContext ) ).After( m_init ).WillOnce( Return( ticket ) );

    m_scene.afterLaunch( m_stream, params );
}

TEST_F( TestPbrtSceneInitialized, cleanupDestroysOptixResources )
{
    EXPECT_CALL( *m_programGroups, cleanup() );

    m_scene.cleanup();
}

TEST_F( TestPbrtSceneInitialized, resolveOneMaterialNotifiesMaterialResolver )
{
    EXPECT_CALL( *m_materialResolver, resolveOneMaterial() ).Times( 1 );

    m_scene.resolveOneMaterial();
}

TEST_F( TestPbrtSceneInitialized, resolveOneGeometryNotifiesGeometryResolver )
{
    EXPECT_CALL( *m_geometryResolver, resolveOneGeometry() ).Times( 1 );

    m_scene.resolveOneGeometry();
}

TEST_F( TestPbrtSceneInitialized, resolvingGeometryUpdatesTopLevel )
{
    expectLaunchPrepareTrueAfter( m_init );
    expectNoMaterialResolvedAfter( m_init );
    EXPECT_CALL( *m_geometryResolver, resolveRequestedProxyGeometries( m_stream, m_fakeContext, _, _ ) )
        .After( m_init )
        .WillOnce( [&]( CUstream, OptixDeviceContext, const FrameStopwatch&, SceneSyncState& sync ) {
            OptixInstance instance{};
            instance.instanceId        = m_fakeMaterialId;
            instance.traversableHandle = m_fakeTriMeshTraversable;
            instance.sbtOffset         = +HitGroupIndex::PROXY_MATERIAL_TRIANGLE;
            sync.topLevelInstances.push_back( instance );
            return true;
        } );
    auto isIAS =
        AllOf( NotNull(),
               hasInstanceBuildInput(
                   0, hasAll( hasNumInstances( 2 ),
                              hasDeviceInstances( hasInstance( 0, hasInstanceTraversable( m_fakeProxyTraversable ) ),
                                                  hasInstance( 1, hasInstanceTraversable( m_fakeTriMeshTraversable ),
                                                               hasInstanceSbtOffset( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE ),
                                                               hasInstanceId( m_fakeMaterialId ) ) ) ) ) );
    expectCreateTopLevelTraversableAfter( isIAS, m_fakeTopLevelTraversable, m_init );

    Params params{};
    m_scene.beforeLaunch( m_stream, params );
}

TEST_F( TestPbrtSceneInitialized, resolvingMaterialUpdatesTopLevel )
{
    expectLaunchPrepareTrueAfter( m_init );
    expectNoGeometryResolvedAfter( m_init );
    EXPECT_CALL( *m_materialResolver, resolveRequestedProxyMaterials( m_stream, _, _ ) )
        .After( m_init )
        .WillOnce( [&]( CUstream, const FrameStopwatch&, SceneSyncState& sync ) {
            OptixInstance instance{};
            instance.instanceId        = m_fakeMaterialId;
            instance.traversableHandle = m_fakeTriMeshTraversable;
            instance.sbtOffset         = +HitGroupIndex::PROXY_MATERIAL_TRIANGLE;
            sync.topLevelInstances.push_back( instance );
            return MaterialResolution::FULL;
        } );
    auto isIAS =
        AllOf( NotNull(),
               hasInstanceBuildInput(
                   0, hasAll( hasNumInstances( 2 ),
                              hasDeviceInstances( hasInstance( 0, hasInstanceTraversable( m_fakeProxyTraversable ) ),
                                                  hasInstance( 1, hasInstanceTraversable( m_fakeTriMeshTraversable ),
                                                               hasInstanceSbtOffset( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE ),
                                                               hasInstanceId( m_fakeMaterialId ) ) ) ) ) );
    expectCreateTopLevelTraversableAfter( isIAS, m_fakeTopLevelTraversable, m_init );

    Params params{};
    m_scene.beforeLaunch( m_stream, params );
}
