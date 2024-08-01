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
#include "MockGeometryLoader.h"
#include "MockImageSource.h"
#include "NullCast.h"
#include "ParamsPrinters.h"
#include "SceneAdapters.h"

#include <DemandTextureCache.h>
#include <ImageSourceFactory.h>
#include <MaterialResolver.h>
#include <Options.h>
#include <Params.h>
#include <PbrtScene.h>
#include <ProgramGroups.h>
#include <Renderer.h>

#include <OptiXToolkit/DemandGeometry/GeometryLoader.h>
#include <OptiXToolkit/DemandGeometry/Mocks/Matchers.h>
#include <OptiXToolkit/DemandGeometry/Mocks/MockDemandLoader.h>
#include <OptiXToolkit/DemandGeometry/Mocks/MockOptix.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneLoader.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <iterator>

using namespace testing;
using namespace demandPbrtScene;
using namespace otk::testing;

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

// For some reason gtest doesn't provide a way to append ExpectationSets, so make this here.
static ExpectationSet& appendTo( ExpectationSet& lhs, const ExpectationSet& rhs )
{
    for( const Expectation& expect : rhs )
    {
        lhs += expect;
    }
    return lhs;
}

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

}  // namespace demandLoading

MATCHER_P2( hasDeviceMaterial, index, material, "" )
{
    if( arg == nullptr )
    {
        *result_listener << "material array pointer is nullptr";
        return false;
    }
    std::vector<PhongMaterial> actualMaterials;
    actualMaterials.resize( index + 1 );
    OTK_ERROR_CHECK( cudaMemcpy( actualMaterials.data(), arg, sizeof( PhongMaterial ) * actualMaterials.size(), cudaMemcpyDeviceToHost ) );
    if( actualMaterials[index] != material )
    {
        *result_listener << "material " << index << " was " << actualMaterials[index] << ", expected " << material;
        return false;
    }

    *result_listener << "material " << index << " matches " << material;
    return true;
}

MATCHER_P2( hasDeviceTriangleNormalPtr, index, normals, "" )
{
    if( arg == nullptr )
    {
        *result_listener << "triangle normals pointer array is nullptr";
        return false;
    }
    std::vector<TriangleNormals*> actualPtrs;
    actualPtrs.resize( index + 1 );
    OTK_ERROR_CHECK( cudaMemcpy( actualPtrs.data(), arg, sizeof( TriangleNormals* ) * actualPtrs.size(), cudaMemcpyDeviceToHost ) );
    if( actualPtrs[index] == nullptr && normals != nullptr )
    {
        *result_listener << "triangle normals pointer at index " << index << " is nullptr";
        return false;
    }
    if( actualPtrs[index] != normals )
    {
        *result_listener << "triangle normals pointer at index " << index << " is " << arg[index] << ", expected " << normals;
        return false;
    }

    *result_listener << "triangle normals pointer at index " << index << " matches " << normals;
    return true;
}

namespace demandLoading {

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

static void identity( float ( &result )[12] )
{
    const float matrix[12]{
        1.0f, 0.0f, 0.0f, 0.0f,  //
        0.0f, 1.0f, 0.0f, 0.0f,  //
        0.0f, 0.0f, 1.0f, 0.0f   //
    };
    std::copy( std::begin( matrix ), std::end( matrix ), std::begin( result ) );
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

class MockDemandTextureCache : public StrictMock<DemandTextureCache>
{
  public:
    ~MockDemandTextureCache() override = default;

    MOCK_METHOD( uint_t, createDiffuseTextureFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( bool, hasDiffuseTextureForFile, ( const std::string& path ), ( const override ) );
    MOCK_METHOD( uint_t, createAlphaTextureFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( bool, hasAlphaTextureForFile, (const std::string&), ( const, override ) );
    MOCK_METHOD( uint_t, createSkyboxTextureFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( bool, hasSkyboxTextureForFile, (const std::string&), ( const, override ) );
    MOCK_METHOD( DemandTextureCacheStatistics, getStatistics, (), ( const override ) );
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

class MockSceneProxy : public StrictMock<SceneProxy>
{
  public:
    ~MockSceneProxy() override = default;

    MOCK_METHOD( uint_t, getPageId, (), ( const, override ) );
    MOCK_METHOD( OptixAabb, getBounds, (), ( const, override ) );
    MOCK_METHOD( bool, isDecomposable, (), ( const, override ) );
    MOCK_METHOD( GeometryInstance, createGeometry, ( OptixDeviceContext, CUstream ), ( override ) );
    MOCK_METHOD( std::vector<SceneProxyPtr>, decompose, ( GeometryLoaderPtr geometryLoader, ProxyFactoryPtr proxyFactory ), ( override ) );
};

class MockProxyFactory : public StrictMock<ProxyFactory>
{
  public:
    ~MockProxyFactory() override = default;

    MOCK_METHOD( SceneProxyPtr, scene, ( GeometryLoaderPtr, SceneDescriptionPtr ), ( override ) );
    MOCK_METHOD( SceneProxyPtr, sceneShape, ( GeometryLoaderPtr, SceneDescriptionPtr, uint_t ), ( override ) );
    MOCK_METHOD( SceneProxyPtr, sceneInstance, ( GeometryLoaderPtr, SceneDescriptionPtr, uint_t ), ( override ) );
    MOCK_METHOD( SceneProxyPtr, sceneInstanceShape, ( GeometryLoaderPtr, SceneDescriptionPtr, uint_t, uint_t ), ( override ) );
    MOCK_METHOD( ProxyFactoryStatistics, getStatistics, (), ( const override ) );
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
using MockDemandTextureCachePtr = std::shared_ptr<MockDemandTextureCache>;
using MockMaterialResolverPtr   = std::shared_ptr<MockMaterialResolver>;
using MockProgramGroupsPtr      = std::shared_ptr<MockProgramGroups>;
using MockProxyFactoryPtr       = std::shared_ptr<MockProxyFactory>;
using MockRendererPtr           = std::shared_ptr<MockRenderer>;
using MockSceneProxyPtr         = std::shared_ptr<MockSceneProxy>;

using AccelBuildOptionsMatcher = Matcher<const OptixAccelBuildOptions*>;
using BuildInputMatcher        = Matcher<const OptixBuildInput*>;

BuildInputMatcher oneInstanceIAS( OptixTraversableHandle instance, uint_t sbtOffset, uint_t instanceId )
{
    return AllOf( NotNull(),
                  hasInstanceBuildInput( 0U, hasAll( hasNumInstances( 1 ),
                                                     hasDeviceInstances( hasInstance( 0, hasInstanceTraversable( instance ),
                                                                                      hasInstanceSbtOffset( sbtOffset ),
                                                                                      hasInstanceId( instanceId ) ) ) ) ) );
}

BuildInputMatcher twoInstanceIAS( OptixTraversableHandle instance1, OptixTraversableHandle instance2, uint_t sbtOffset1, uint_t instanceId1 = 0 )
{
    return AllOf( NotNull(),
                  hasInstanceBuildInput( 0, hasAll( hasNumInstances( 2 ),
                                                    hasDeviceInstances( hasInstance( 0U, hasInstanceTraversable( instance1 ) ),
                                                                        hasInstance( 1U, hasInstanceTraversable( instance2 ),
                                                                                     hasInstanceSbtOffset( sbtOffset1 ),
                                                                                     hasInstanceId( instanceId1 ) ) ) ) ) );
}

// This was needed to satisfy gcc instead of constructing from a brace initializer list.
Options testOptions()
{
    Options options{};
    options.program   = "DemandPbrtScene";
    options.sceneFile = "test.pbrt";
    options.outFile   = "out.png";
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
    MockDemandTextureCachePtr m_demandTextureCache{ std::make_shared<MockDemandTextureCache>() };
    MockProxyFactoryPtr       m_proxyFactory{ std::make_shared<MockProxyFactory>() };
    MockDemandLoaderPtr       m_demandLoader{ std::make_shared<StrictMockDemandLoader>() };
    MockGeometryLoaderPtr     m_geometryLoader{ std::make_shared<MockGeometryLoader>() };
    MockProgramGroupsPtr      m_programGroups{ std::make_shared<MockProgramGroups>() };
    MockMaterialResolverPtr   m_materialResolver{ std::make_shared<MockMaterialResolver>() };
    MockRendererPtr           m_renderer{ std::make_shared<MockRenderer>() };
    Options                   m_options{ testOptions() };
    // clang-format off
    PbrtScene m_scene{ m_options, m_sceneLoader, m_demandTextureCache, m_proxyFactory, m_demandLoader, m_geometryLoader, m_programGroups, m_materialResolver, m_renderer };
    // clang-format on
    SceneDescriptionPtr          m_sceneDesc{ std::make_shared<otk::pbrt::SceneDescription>() };
    OptixAabb                    m_sceneBounds{ -1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f };
    MockSceneProxyPtr            m_mockSceneProxy{ std::make_shared<MockSceneProxy>() };
    uint_t                       m_scenePageId{ 6646U };
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
    expect += EXPECT_CALL( *m_mockSceneProxy, getPageId() ).WillOnce( Return( m_scenePageId ) );
    // TODO: Determine why adding this expectation to the set causes a dangling reference to m_mockSceneProxy
    /*expect +=*/EXPECT_CALL( *m_proxyFactory, scene( _, _ ) ).WillOnce( Return( m_mockSceneProxy ) );
    expect += EXPECT_CALL( *m_programGroups, initialize() );
    // This getter can be called anytime in any order.
    EXPECT_CALL( *m_renderer, getDeviceContext() ).WillRepeatedly( Return( m_fakeContext ) );
    expect += EXPECT_CALL( *m_geometryLoader, setSbtIndex( _ ) ).Times( AtLeast( 1 ) );
    expect += EXPECT_CALL( *m_geometryLoader, copyToDeviceAsync( m_stream ) ).Times( AtLeast( 1 ) );
    expect += EXPECT_CALL( *m_geometryLoader, createTraversable( _, _ ) ).WillOnce( Return( m_fakeProxyTraversable ) );
    // This getter can be called anytime in any order.
    EXPECT_CALL( *m_renderer, getPipelineCompileOptions() ).WillRepeatedly( ReturnRef( m_pipelineCompileOptions ) );
    expect += expectAccelComputeMemoryUsage( _, _ );
    expect += expectAccelBuild( _, _, m_fakeTopLevelTraversable );
    return expect;
}

ExpectationSet TestPbrtScene::expectInitialize()
{
    ExpectationSet state = expectInitializeCreatesOptixState();
    state += EXPECT_CALL( *m_renderer, setLookAt( _ ) );
    state += EXPECT_CALL( *m_renderer, setCamera( _ ) );
    return state;
}

class TestPbrtSceneInitialized : public TestPbrtScene
{
  public:
    ~TestPbrtSceneInitialized() override = default;

  protected:
    void SetUp() override;

    ExpectationSet expectCreateTopLevelTraversable( const BuildInputMatcher& buildInput,
                                                    OptixTraversableHandle   result,
                                                    const ExpectationSet&    before );
    Expectation    expectRequestedProxyIdsAfter( std::initializer_list<uint_t> pageIds, const ExpectationSet& before );
    Expectation expectClearRequestedProxyIdsAfter( const ExpectationSet& before );
    Expectation expectProxyMaterialsResolvedAfter( MaterialResolution resolution, const ExpectationSet&before);
    Expectation expectGeometryLoaderGetContextAfter( const ExpectationSet& before );
    Expectation expectLaunchPrepareTrueAfter( const ExpectationSet& before );
    Expectation expectProxyDecomposableAfter( MockSceneProxyPtr proxy, bool decomposable, const ExpectationSet& before );
    Expectation expectProxyRemovedAfter( uint_t pageId, const ExpectationSet& before );
    Expectation expectSceneDecomposedAfterInitTo( const std::vector<SceneProxyPtr>& shapeProxies );
    Expectation expectGeometryLoaderCopiedToDeviceAfter( const ExpectationSet& first );
    Expectation expectGeometryLoaderCreateTraversableAfter( OptixTraversableHandle traversable, const ExpectationSet& before );
    Expectation expectProxyCreateGeometryAfter( MockSceneProxyPtr proxy, const GeometryInstance& geometry, const ExpectationSet& before );
    Expectation expectSceneProxyCreateGeometryAfter( const GeometryInstance& geometry, const ExpectationSet& before );
    GeometryInstance proxyMaterialTriMeshGeometry();
    GeometryInstance proxyMaterialSphereGeometry();
    ExpectationSet   expectNoGeometryResolvedAfter( const ExpectationSet& before );
    Expectation expectNoMaterialResolvedAfter( const ExpectationSet& before );
    ExpectationSet   expectNoTopLevelASBuildAfter( const ExpectationSet& before );
    void             setDistantLightOnSceneDescription();
    void             setInfiniteLightOnSceneDescription();

    ExpectationSet         m_init;
    OptixTraversableHandle m_fakeUpdatedProxyTraversable{ 0xf00dbad2U };
    OptixTraversableHandle m_fakeTriMeshTraversable{ 0xbeefbeefU };
    OptixTraversableHandle m_fakeSphereTraversable{ 0x11110000U };
    uint_t                 m_fakeMaterialId{ 666U };
    PhongMaterial          m_realizedMaterial{
        make_float3( 0.1f, 0.2f, 0.3f ),     // Ka
        make_float3( 0.4f, 0.5f, 0.6f ),     // Kd
        make_float3( 0.7f, 0.8f, 0.9f ),     // Ks
        make_float3( 0.11f, 0.22f, 0.33f ),  // Kr
        128.0f,                              // phongExp
        MaterialFlags::NONE,                 // flags
        0U,                                  // alphaTextureId
    };
    DirectionalLight m_expectedDirectionalLight{};
    InfiniteLight    m_expectedInfiniteLight{};
    TriangleNormals* m_fakeTriangleNormals{ reinterpret_cast<TriangleNormals*>( static_cast<std::uintptr_t>( 0xfadefadeU ) ) };
    TriangleUVs* m_fakeTriangleUVs{ reinterpret_cast<TriangleUVs*>( static_cast<std::uintptr_t>( 0xfadef00dU ) ) };
};

void TestPbrtSceneInitialized::SetUp()
{
    TestPbrtScene::SetUp();
    m_init = expectInitialize();
    m_scene.initialize( m_stream );
}

ExpectationSet TestPbrtSceneInitialized::expectCreateTopLevelTraversable( const BuildInputMatcher& buildInput,
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

Expectation TestPbrtSceneInitialized::expectRequestedProxyIdsAfter( std::initializer_list<uint_t> pageIds, const ExpectationSet& before )
{
    return EXPECT_CALL( *m_geometryLoader, requestedProxyIds() ).After( before ).WillOnce( Return( std::vector<uint_t>{ pageIds } ) );
}

Expectation TestPbrtSceneInitialized::expectClearRequestedProxyIdsAfter( const ExpectationSet& before )
{
    return EXPECT_CALL( *m_geometryLoader, clearRequestedProxyIds() ).After( before );
}

Expectation TestPbrtSceneInitialized::expectProxyMaterialsResolvedAfter( MaterialResolution resolution, const ExpectationSet& before )
{
    return EXPECT_CALL( *m_materialResolver, resolveRequestedProxyMaterials( _, _, _ ) ).After( before ).WillOnce( Return( resolution ) );
}

Expectation TestPbrtSceneInitialized::expectGeometryLoaderGetContextAfter( const ExpectationSet& before )
{
    return EXPECT_CALL( *m_geometryLoader, getContext() ).After( before ).WillRepeatedly( Return( m_demandGeomContext ) );
}

Expectation TestPbrtSceneInitialized::expectLaunchPrepareTrueAfter( const ExpectationSet& before )
{
    return EXPECT_CALL( *m_demandLoader, launchPrepare( m_stream, _ ) )
        .After( before )
        .WillOnce( DoAll( SetArgReferee<1>( m_demandLoadContext ), Return( true ) ) );
}

Expectation TestPbrtSceneInitialized::expectProxyDecomposableAfter( MockSceneProxyPtr proxy, bool decomposable, const ExpectationSet& before )
{
    return EXPECT_CALL( *proxy, isDecomposable() ).After( before ).WillOnce( Return( decomposable ) );
}

Expectation TestPbrtSceneInitialized::expectProxyRemovedAfter( uint_t pageId, const ExpectationSet& before )
{
    return EXPECT_CALL( *m_geometryLoader, remove( pageId ) ).Times( 1 ).After( before );
}

Expectation TestPbrtSceneInitialized::expectSceneDecomposedAfterInitTo( const std::vector<SceneProxyPtr>& shapeProxies )
{
    return EXPECT_CALL( *m_mockSceneProxy, decompose( static_cast<GeometryLoaderPtr>( m_geometryLoader ),
                                                      static_cast<ProxyFactoryPtr>( m_proxyFactory ) ) )
        .After( m_init )
        .WillOnce( Return( shapeProxies ) );
}

Expectation TestPbrtSceneInitialized::expectGeometryLoaderCopiedToDeviceAfter( const ExpectationSet& first )
{
    return EXPECT_CALL( *m_geometryLoader, copyToDeviceAsync( m_stream ) ).After( first );
}

Expectation TestPbrtSceneInitialized::expectGeometryLoaderCreateTraversableAfter( OptixTraversableHandle traversable,
                                                                                  const ExpectationSet&  before )
{
    return EXPECT_CALL( *m_geometryLoader, createTraversable( m_fakeContext, m_stream ) ).After( before ).WillOnce( Return( traversable ) );
}

Expectation TestPbrtSceneInitialized::expectProxyCreateGeometryAfter( MockSceneProxyPtr       proxy,
                                                                      const GeometryInstance& geometry,
                                                                      const ExpectationSet&   before )
{
    return EXPECT_CALL( *proxy, createGeometry( m_fakeContext, m_stream ) ).After( before ).WillOnce( Return( geometry ) );
}

Expectation TestPbrtSceneInitialized::expectSceneProxyCreateGeometryAfter( const GeometryInstance& geometry, const ExpectationSet& before )
{
    return expectProxyCreateGeometryAfter( m_mockSceneProxy, geometry, before );
}

GeometryInstance TestPbrtSceneInitialized::proxyMaterialTriMeshGeometry()
{
    GeometryInstance geometry{};
    identity( geometry.instance.transform );
    geometry.instance.traversableHandle = m_fakeTriMeshTraversable;
    geometry.instance.sbtOffset         = +HitGroupIndex::PROXY_MATERIAL_TRIANGLE;
    geometry.primitive                  = GeometryPrimitive::TRIANGLE;
    geometry.material                   = m_realizedMaterial;
    geometry.normals                    = m_fakeTriangleNormals;
    geometry.uvs                        = m_fakeTriangleUVs;
    return geometry;
}

GeometryInstance TestPbrtSceneInitialized::proxyMaterialSphereGeometry()
{
    GeometryInstance geometry{};
    identity( geometry.instance.transform );
    geometry.instance.traversableHandle = m_fakeSphereTraversable;
    geometry.instance.sbtOffset         = +HitGroupIndex::PROXY_MATERIAL_SPHERE;
    geometry.primitive                  = GeometryPrimitive::SPHERE;
    geometry.material                   = m_realizedMaterial;
    return geometry;
}

ExpectationSet TestPbrtSceneInitialized::expectNoGeometryResolvedAfter( const ExpectationSet& before )
{
    ExpectationSet expect;
    expect += EXPECT_CALL( *m_geometryLoader, requestedProxyIds() ).Times( 0 ).After( before );
    expect += EXPECT_CALL( *m_mockSceneProxy, isDecomposable() ).Times( 0 ).After( before );
    expect += EXPECT_CALL( *m_geometryLoader, remove( m_scenePageId ) ).Times( 0 ).After( before );
    expect += EXPECT_CALL( *m_geometryLoader, copyToDeviceAsync( m_stream ) ).Times( 0 ).After( before );
    expect += EXPECT_CALL( *m_geometryLoader, createTraversable( m_fakeContext, m_stream ) ).Times( 0 ).After( before );
    expect += EXPECT_CALL( *m_mockSceneProxy, createGeometry( m_fakeContext, m_stream ) ).Times( 0 ).After( before );
    return expect;
}

Expectation TestPbrtSceneInitialized::expectNoMaterialResolvedAfter( const ExpectationSet& before )
{
    return expectProxyMaterialsResolvedAfter( MaterialResolution::NONE, before );
}

ExpectationSet TestPbrtSceneInitialized::expectNoTopLevelASBuildAfter( const ExpectationSet& before )
{
    ExpectationSet expect;
    expect += EXPECT_CALL( m_optix, accelComputeMemoryUsage( m_fakeContext, _, _, _, _ ) ).Times( 0 ).After( before );
    expect +=
        EXPECT_CALL( m_optix, accelBuild( m_fakeContext, m_stream, _, _, _, _, _, _, _, _, nullptr, 0 ) ).Times( 0 ).After( before );
    return expect;
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

TEST_F( TestPbrtScene, initializeCreatesOptixResourcesForLoadedScene )
{
    EXPECT_CALL( *m_sceneLoader, parseFile( m_options.sceneFile ) ).Times( 1 ).WillOnce( Return( m_sceneDesc ) );
    EXPECT_CALL( *m_proxyFactory, scene( static_cast<GeometryLoaderPtr>( m_geometryLoader ), m_sceneDesc ) ).WillOnce( Return( m_mockSceneProxy ) );
    EXPECT_CALL( *m_mockSceneProxy, getPageId() ).WillOnce( Return( m_scenePageId ) );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).WillRepeatedly( Return( m_fakeContext ) );
    EXPECT_CALL( *m_renderer, getPipelineCompileOptions() ).WillRepeatedly( ReturnRef( m_pipelineCompileOptions ) );
    EXPECT_CALL( *m_programGroups, initialize() );
    EXPECT_CALL( *m_geometryLoader, setSbtIndex( _ ) ).Times( AtLeast( 1 ) );
    EXPECT_CALL( *m_geometryLoader, copyToDeviceAsync( m_stream ) ).Times( 1 );
    EXPECT_CALL( *m_geometryLoader, createTraversable( m_fakeContext, m_stream ) ).WillOnce( Return( m_fakeProxyTraversable ) );
    auto isIAS =
        AllOf( NotNull(),
               hasInstanceBuildInput( 0, hasAll( hasNumInstances( 1 ),
                                                 hasDeviceInstances( hasInstance( 0, hasInstanceTraversable( m_fakeProxyTraversable ),
                                                                                  hasInstanceSbtOffset( +HitGroupIndex::PROXY_GEOMETRY ),
                                                                                  hasInstanceId( 0 ) ) ) ) ) );
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
    EXPECT_CALL( *m_geometryLoader, getContext() ).WillRepeatedly( Return( m_demandGeomContext ) );
    expectLaunchPrepareTrueAfter( m_init );
    expectRequestedProxyIdsAfter( {}, m_init );
    expectClearRequestedProxyIdsAfter( m_init );
    expectNoMaterialResolvedAfter( m_init );

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

TEST_F( TestPbrtSceneInitialized, beforeLaunchSetsDirectionalLightsInParams )
{
    setDistantLightOnSceneDescription();
    EXPECT_CALL( *m_geometryLoader, getContext() ).WillRepeatedly( Return( m_demandGeomContext ) );
    expectLaunchPrepareTrueAfter( m_init );
    expectRequestedProxyIdsAfter( {}, m_init );
    expectClearRequestedProxyIdsAfter( m_init );
    expectNoMaterialResolvedAfter( m_init );

    Params params{};
    m_scene.beforeLaunch( m_stream, params );

    ASSERT_EQ( 1, params.numDirectionalLights );
    EXPECT_THAT( params.directionalLights, hasDeviceDirectionalLight( m_expectedDirectionalLight ) );
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

TEST_F( TestPbrtSceneInitialized, beforeLaunchSetsInfiniteLightsInParams )
{
    setInfiniteLightOnSceneDescription();
    EXPECT_CALL( *m_geometryLoader, getContext() ).WillRepeatedly( Return( m_demandGeomContext ) );
    expectLaunchPrepareTrueAfter( m_init );
    expectRequestedProxyIdsAfter( {}, m_init );
    expectClearRequestedProxyIdsAfter( m_init );
    expectNoMaterialResolvedAfter( m_init );

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
    EXPECT_CALL( *m_geometryLoader, getContext() ).WillRepeatedly( Return( m_demandGeomContext ) );
    expectLaunchPrepareTrueAfter( m_init );
    expectRequestedProxyIdsAfter( {}, m_init );
    expectClearRequestedProxyIdsAfter( m_init );
    expectNoMaterialResolvedAfter( m_init );
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

TEST_F( TestPbrtSceneInitialized, firstLaunchResolvesSceneProxyToSingleTriMeshWithProxyMaterial )
{
    expectGeometryLoaderGetContextAfter( m_init );
    expectRequestedProxyIdsAfter( { m_scenePageId }, m_init );
    expectClearRequestedProxyIdsAfter( m_init );
    expectNoMaterialResolvedAfter( m_init );
    expectProxyDecomposableAfter( m_mockSceneProxy, false, m_init );
    expectProxyRemovedAfter( m_scenePageId, m_init );
    OptixTraversableHandle updatedProxyTravHandle{ 0xf00d13f00d13U };
    expectGeometryLoaderCopiedToDeviceAfter( m_init );
    expectGeometryLoaderCreateTraversableAfter( updatedProxyTravHandle, m_init );
    const GeometryInstance proxyMaterialGeom = proxyMaterialTriMeshGeometry();
    expectSceneProxyCreateGeometryAfter( proxyMaterialGeom, m_init );
    const auto isTopLevelIAS = twoInstanceIAS( updatedProxyTravHandle, proxyMaterialGeom.instance.traversableHandle,
                                               +HitGroupIndex::PROXY_MATERIAL_TRIANGLE, m_fakeMaterialId );
    expectCreateTopLevelTraversable( isTopLevelIAS, m_fakeTopLevelTraversable, m_init );
    expectLaunchPrepareTrueAfter( m_init );
    EXPECT_CALL( *m_materialResolver, resolveMaterialForGeometry( m_scenePageId, _, _ ) )
        .After( m_init )
        .WillOnce( [=]( uint_t, SceneGeometry& geom, SceneSyncState& syncState ) {
            const uint_t materialId{ m_fakeMaterialId };
            const uint_t instanceId{ materialId };  // use the proxy material id as the instance id
            geom.materialId                   = materialId;
            geom.instance.instance.instanceId = instanceId;
            geom.instanceIndex                = syncState.topLevelInstances.size();
            syncState.topLevelInstances.push_back( geom.instance.instance );
            return false;
        } );

    Params      params{};
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    EXPECT_EQ( 1, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 1, stats.numGeometriesRealized );
}

TEST_F( TestPbrtSceneInitialized, firstLaunchResolvesNoGeometryUntilOneShotFired )
{
    m_options.oneShotGeometry = true;
    expectGeometryLoaderGetContextAfter( m_init );
    expectNoGeometryResolvedAfter( m_init );
    expectNoTopLevelASBuildAfter( m_init );
    expectLaunchPrepareTrueAfter( m_init );
    expectNoMaterialResolvedAfter( m_init );

    Params      params{};
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    EXPECT_EQ( 0, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 0, stats.numGeometriesRealized );
}

TEST_F( TestPbrtSceneInitialized, resolveOneGeometryAfterOneShotFired )
{
    m_options.oneShotGeometry = true;
    ExpectationSet first;
    first += expectNoMaterialResolvedAfter( m_init );
    first += expectGeometryLoaderGetContextAfter( m_init );
    first += expectRequestedProxyIdsAfter( { m_scenePageId }, m_init );
    first += expectClearRequestedProxyIdsAfter( m_init );
    first += expectProxyDecomposableAfter( m_mockSceneProxy, false, m_init );
    first += expectProxyRemovedAfter( m_scenePageId, m_init );
    const OptixTraversableHandle updatedProxyTravHandle{ 0xf00d13f00d13U };
    first += expectGeometryLoaderCopiedToDeviceAfter( m_init );
    first += expectGeometryLoaderCreateTraversableAfter( updatedProxyTravHandle, m_init );
    const GeometryInstance proxyMaterialGeom = proxyMaterialTriMeshGeometry();
    first += expectSceneProxyCreateGeometryAfter( proxyMaterialGeom, m_init );
    auto isTopLevelIAS =
        AllOf( NotNull(),
               hasInstanceBuildInput(
                   0, hasAll( hasNumInstances( 2 ),
                              hasDeviceInstances( hasInstance( 0, hasInstanceTraversable( updatedProxyTravHandle ) ),
                                                  hasInstance( 1, hasInstanceTraversable( proxyMaterialGeom.instance.traversableHandle ),
                                                               hasInstanceSbtOffset( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE ),
                                                               hasInstanceId( m_fakeMaterialId ) ) ) ) ) );
    appendTo( first, expectCreateTopLevelTraversable( isTopLevelIAS, m_fakeTopLevelTraversable, m_init ) );
    first += expectLaunchPrepareTrueAfter( m_init );
    expectGeometryLoaderGetContextAfter( first );
    expectNoGeometryResolvedAfter( first );
    expectNoMaterialResolvedAfter( first );
    expectNoTopLevelASBuildAfter( first );
    expectLaunchPrepareTrueAfter( first );
    EXPECT_CALL( *m_materialResolver, resolveMaterialForGeometry( m_scenePageId, _, _ ) )
        .After( m_init )
        .WillOnce( [=]( uint_t, SceneGeometry& geom, SceneSyncState& syncState ) {
            const uint_t materialId{ m_fakeMaterialId };
            const uint_t instanceId{ materialId };  // use the proxy material id as the instance id
            geom.materialId                   = materialId;
            geom.instance.instance.instanceId = instanceId;
            geom.instanceIndex                = syncState.topLevelInstances.size();
            syncState.topLevelInstances.push_back( geom.instance.instance );
            return false;
        } );

    Params params{};
    m_scene.resolveOneGeometry();
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    EXPECT_EQ( 1, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 1, stats.numGeometriesRealized );
}

MockSceneProxyPtr createGeometryProxyAfter( uint_t pageId, ExpectationSet& before )
{
    auto proxy = std::make_shared<MockSceneProxy>();
    EXPECT_CALL( *proxy, getPageId() ).After( before ).WillRepeatedly( Return( pageId ) );
    return proxy;
}

TEST_F( TestPbrtSceneInitialized, firstLaunchResolvesDecomposableSceneToShapeProxies )
{
    expectGeometryLoaderGetContextAfter( m_init );
    expectLaunchPrepareTrueAfter( m_init );
    expectRequestedProxyIdsAfter( { m_scenePageId }, m_init );
    expectClearRequestedProxyIdsAfter( m_init );
    expectNoMaterialResolvedAfter( m_init );
    expectProxyDecomposableAfter( m_mockSceneProxy, true, m_init );
    expectProxyRemovedAfter( m_scenePageId, m_init );
    std::vector<SceneProxyPtr> shapeProxies{ ( createGeometryProxyAfter( 1111, m_init ) ),
                                             ( createGeometryProxyAfter( 2222, m_init ) ) };
    expectSceneDecomposedAfterInitTo( shapeProxies );
    OptixTraversableHandle updatedProxyTraversable{ 0xf00d13f00d13U };
    expectGeometryLoaderCopiedToDeviceAfter( m_init );
    expectGeometryLoaderCreateTraversableAfter( updatedProxyTraversable, m_init );
    expectCreateTopLevelTraversable( oneInstanceIAS( updatedProxyTraversable, +HitGroupIndex::PROXY_GEOMETRY, 0 ),
                                     m_fakeTopLevelTraversable, m_init );

    Params      params{};
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    EXPECT_EQ( 1, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 0, stats.numGeometriesRealized );
    // gmock is holding onto these objects internally somehow
    for( auto& proxy : shapeProxies )
    {
        Mock::AllowLeak( proxy.get() );
    }
    Mock::AllowLeak( createGeometryProxyAfter( 2222, m_init ).get() );
    Mock::AllowLeak( m_geometryLoader.get() );
    Mock::AllowLeak( m_mockSceneProxy.get() );
    Mock::AllowLeak( m_proxyFactory.get() );
}

TEST_F( TestPbrtSceneInitialized, resolveSceneToSphereAndTriMesh )
{
    // first launch resolves scene proxy to child proxies
    ExpectationSet first;
    first += expectGeometryLoaderGetContextAfter( m_init );
    first += expectLaunchPrepareTrueAfter( m_init );
    first += expectRequestedProxyIdsAfter( { m_scenePageId }, m_init );
    first += expectClearRequestedProxyIdsAfter( m_init );
    first += expectNoMaterialResolvedAfter( m_init );
    first += expectProxyDecomposableAfter( m_mockSceneProxy, true, m_init );
    first += expectProxyRemovedAfter( m_scenePageId, m_init );
    const uint_t               spherePageId{ 1111 };
    const uint_t               triMeshPageId{ 2222 };
    MockSceneProxyPtr          sphereProxy  = createGeometryProxyAfter( spherePageId, m_init );
    MockSceneProxyPtr          triMeshProxy = createGeometryProxyAfter( triMeshPageId, m_init );
    std::vector<SceneProxyPtr> shapeProxies{ sphereProxy, triMeshProxy };
    first += expectSceneDecomposedAfterInitTo( shapeProxies );
    OptixTraversableHandle proxyTraversable2{ 0xf00d13f00d13U };
    first += expectGeometryLoaderCopiedToDeviceAfter( m_init );
    first += expectGeometryLoaderCreateTraversableAfter( proxyTraversable2, m_init );
    auto isTopLevelIAS =
        AllOf( NotNull(), hasInstanceBuildInput( 0, hasDeviceInstances( hasInstance( 0, hasInstanceSbtOffset( +HitGroupIndex::PROXY_GEOMETRY ),
                                                                                     hasInstanceId( 0 ) ) ) ) );
    auto isFirstLaunchIAS =
        AllOf( isTopLevelIAS,
               hasInstanceBuildInput( 0, hasAll( hasNumInstances( 1 ),
                                                 hasDeviceInstances( hasInstance( 0, hasInstanceTraversable( proxyTraversable2 ) ) ) ) ) );
    appendTo( first, expectCreateTopLevelTraversable( isFirstLaunchIAS, m_fakeTopLevelTraversable, m_init ) );
    // second launch resolves proxies to real geometry with proxy materials
    expectGeometryLoaderGetContextAfter( first );
    expectLaunchPrepareTrueAfter( first );
    expectRequestedProxyIdsAfter( { spherePageId, triMeshPageId }, first );
    expectClearRequestedProxyIdsAfter( first );
    expectNoMaterialResolvedAfter( first );
    expectProxyDecomposableAfter( sphereProxy, false, first );
    expectProxyDecomposableAfter( triMeshProxy, false, first );
    expectProxyRemovedAfter( spherePageId, first );
    expectProxyRemovedAfter( triMeshPageId, first );
    ExpectationSet createSphere = expectProxyCreateGeometryAfter( sphereProxy, proxyMaterialSphereGeometry(), first );
    ExpectationSet createTriMesh = expectProxyCreateGeometryAfter( triMeshProxy, proxyMaterialTriMeshGeometry(), first );
    EXPECT_CALL( *m_materialResolver, resolveMaterialForGeometry( spherePageId, _, _ ) )
        .After( first )
        .WillOnce( [=]( uint_t, SceneGeometry& geom, SceneSyncState& syncState ) {
            const uint_t materialId{ m_fakeMaterialId };
            const uint_t instanceId{ materialId };  // use the proxy material id as the instance id
            geom.materialId                   = materialId;
            geom.instance.instance.instanceId = instanceId;
            geom.instanceIndex                = syncState.topLevelInstances.size();
            syncState.topLevelInstances.push_back( geom.instance.instance );
            return false;
        } );
    EXPECT_CALL( *m_materialResolver, resolveMaterialForGeometry( triMeshPageId, _, _ ) )
        .After( first )
        .WillOnce( [=]( uint_t, SceneGeometry& geom, SceneSyncState& syncState ) {
            const uint_t materialId{ m_fakeMaterialId };
            const uint_t instanceId{ materialId };  // use the proxy material id as the instance id
            geom.materialId                   = materialId;
            geom.instance.instance.instanceId = instanceId;
            geom.instanceIndex                = syncState.topLevelInstances.size();
            syncState.topLevelInstances.push_back( geom.instance.instance );
            return false;
        } );
    expectGeometryLoaderCopiedToDeviceAfter( first );
    OptixTraversableHandle proxyTraversable3{ 0xbadd1ebadd1eU };
    expectGeometryLoaderCreateTraversableAfter( proxyTraversable3, first );
    auto isSecondLaunchIAS = AllOf(
        isTopLevelIAS,
        hasInstanceBuildInput(
            0, hasAll( hasNumInstances( 3 ),
                       hasDeviceInstances( hasInstance( 0, hasInstanceTraversable( proxyTraversable3 ),
                                                        hasInstanceSbtOffset( +HitGroupIndex::PROXY_GEOMETRY ) ),
                                           hasInstance( 1, hasInstanceTraversable( m_fakeSphereTraversable ),
                                                        hasInstanceSbtOffset( +HitGroupIndex::PROXY_MATERIAL_SPHERE ) ),
                                           hasInstance( 2, hasInstanceTraversable( m_fakeTriMeshTraversable ),
                                                        hasInstanceSbtOffset( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE ) ) ) ) ) );
    expectCreateTopLevelTraversable( isSecondLaunchIAS, m_fakeTopLevelTraversable, first );
    // third launch resolves proxy materials to real materials

    Params params{};
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    const Stats stats = m_scene.getStatistics();

    EXPECT_EQ( 3, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 2, stats.numGeometriesRealized );
    // gmock is holding onto these objects internally somehow
    Mock::AllowLeak( sphereProxy.get() );
    Mock::AllowLeak( triMeshProxy.get() );
    Mock::AllowLeak( m_geometryLoader.get() );
    Mock::AllowLeak( m_mockSceneProxy.get() );
    Mock::AllowLeak( m_proxyFactory.get() );
}

namespace {

class TestPbrtSceneFirstLaunch : public TestPbrtSceneInitialized
{
  public:
    ~TestPbrtSceneFirstLaunch() override = default;

  protected:
    void SetUp() override;

    virtual GeometryInstance sceneGeometry() = 0;
    void                     expectFirstLaunchResolvesSceneToGeometry();

    ExpectationSet m_first;
};

void TestPbrtSceneFirstLaunch::SetUp()
{
    TestPbrtSceneInitialized::SetUp();
    expectFirstLaunchResolvesSceneToGeometry();
}

void TestPbrtSceneFirstLaunch::expectFirstLaunchResolvesSceneToGeometry()
{
    m_first += expectGeometryLoaderGetContextAfter( m_init );
    m_first += expectRequestedProxyIdsAfter( { m_scenePageId }, m_init );
    m_first += expectClearRequestedProxyIdsAfter( m_init );
    m_first += expectNoMaterialResolvedAfter( m_init );
    m_first += expectProxyDecomposableAfter( m_mockSceneProxy, false, m_init );
    m_first += expectProxyRemovedAfter( m_scenePageId, m_init );
    m_first += expectGeometryLoaderCopiedToDeviceAfter( m_init );
    m_first += expectGeometryLoaderCreateTraversableAfter( m_fakeUpdatedProxyTraversable, m_init );
    m_first += expectSceneProxyCreateGeometryAfter( sceneGeometry(), m_init );
    m_first += EXPECT_CALL( *m_materialResolver, resolveMaterialForGeometry( m_scenePageId, _, _ ) )
                   .After( m_init )
                   .WillOnce( [=]( uint_t, SceneGeometry& geom, SceneSyncState& syncState ) {
                       const uint_t materialId{ m_fakeMaterialId };
                       const uint_t instanceId{ materialId };  // use the proxy material id as the instance id
                       geom.materialId                   = materialId;
                       geom.instance.instance.instanceId = instanceId;
                       geom.instanceIndex                = syncState.topLevelInstances.size();
                       syncState.topLevelInstances.push_back( geom.instance.instance );
                       return false;
                   } );
    appendTo( m_first, expectCreateTopLevelTraversable( _, m_fakeTopLevelTraversable, m_init ) );
    m_first += expectLaunchPrepareTrueAfter( m_init );
}

}  // namespace

namespace {

class TestPbrtSceneTriMesh : public TestPbrtSceneFirstLaunch
{
  public:
    ~TestPbrtSceneTriMesh() override = default;

  protected:
    GeometryInstance sceneGeometry() override { return proxyMaterialTriMeshGeometry(); }
};

}  // namespace

TEST_F( TestPbrtSceneTriMesh, resolvesNoMaterialsUntilOneShotFired )
{
    m_options.oneShotMaterial = true;
    expectRequestedProxyIdsAfter( {}, m_first );
    expectClearRequestedProxyIdsAfter( m_first );
    expectNoMaterialResolvedAfter( m_first );
    expectNoTopLevelASBuildAfter( m_first );
    expectGeometryLoaderGetContextAfter( m_first );
    expectLaunchPrepareTrueAfter( m_first );

    Params params{};
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    EXPECT_THAT( params.realizedMaterials, Not( hasDeviceMaterial( 0, m_realizedMaterial ) ) );
    EXPECT_THAT( params.instanceNormals, Not( hasDeviceTriangleNormalPtr( 0, m_fakeTriangleNormals ) ) );
    EXPECT_EQ( 1, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 1, stats.numGeometriesRealized );
}
