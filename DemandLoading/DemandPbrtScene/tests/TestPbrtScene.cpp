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

#include <PbrtScene.h>

#include "Matchers.h"
#include "MockGeometryLoader.h"
#include "MockImageSource.h"
#include "NullCast.h"
#include "ParamsPrinters.h"
#include "SceneAdapters.h"

#include <DemandTextureCache.h>
#include <ImageSourceFactory.h>
#include <Options.h>
#include <Params.h>
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

#include "Matchers.h"

using namespace testing;
using namespace demandPbrtScene;
using namespace otk::testing;

using P3 = pbrt::Point3f;
using V3 = pbrt::Vector3f;
using B3 = pbrt::Bounds3f;

using Stats = SceneStatistics;

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

MATCHER( hasModuleTypeTriangle, "" )
{
    const bool result = arg->builtinISModuleType == OPTIX_PRIMITIVE_TYPE_TRIANGLE;
    if( !result )
    {
        *result_listener << "module has type " << arg->builtinISModuleType
                         << ", expected OPTIX_PRIMITIVE_TYPE_TRIANGLE (" << OPTIX_PRIMITIVE_TYPE_TRIANGLE << ")";
    }
    else
    {
        *result_listener << "module has type OPTIX_PRIMITIVE_TYPE_TRIANGLE (" << OPTIX_PRIMITIVE_TYPE_TRIANGLE << ")";
    }
    return result;
}

MATCHER( hasModuleTypeSphere, "" )
{
    const bool result = arg->builtinISModuleType == OPTIX_PRIMITIVE_TYPE_SPHERE;
    if( !result )
    {
        *result_listener << "module has type " << arg->builtinISModuleType << ", expected OPTIX_PRIMITIVE_TYPE_SPHERE ("
                         << OPTIX_PRIMITIVE_TYPE_SPHERE << ")";
    }
    else
    {
        *result_listener << "module has type OPTIX_PRIMITIVE_TYPE_SPHERE (" << OPTIX_PRIMITIVE_TYPE_SPHERE << ")";
    }
    return result;
}

MATCHER( allowsRandomVertexAccess, "" )
{
    const bool result = ( arg->buildFlags & OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS ) == OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    const char fill = result_listener->stream()->fill();
    if( !result )
    {
        *result_listener << "builtin IS module options build flags " << std::dec << arg->buildFlags << " (0x"
                         << std::hex << std::setw( 2 * sizeof( arg->buildFlags ) ) << std::setfill( '0' )
                         << arg->buildFlags << ") don't set OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS (0x" << std::hex
                         << std::setw( 2 * sizeof( arg->buildFlags ) ) << std::setfill( '0' )
                         << OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS << ")" << std::setfill( fill );
    }
    else
    {
        *result_listener << "builtin IS module options build flags " << std::dec << arg->buildFlags << " (0x"
                         << std::hex << std::setw( 2 * sizeof( arg->buildFlags ) ) << std::setfill( '0' )
                         << arg->buildFlags << ") set OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS (0x" << std::hex
                         << std::setw( 2 * sizeof( arg->buildFlags ) ) << std::setfill( '0' )
                         << OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS << ")" << std::setfill( fill );
    }
    return result;
}

MATCHER_P( hasProgramGroupCount, count, "" )
{
    if( arg.size() != count )
    {
        *result_listener << "program group count " << arg.size() << ", expected " << count;
        return false;
    }

    *result_listener << "has program group count " << count;
    return true;
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

MATCHER_P2( hasDevicePartialMaterial, index, material, "" )
{
    if( arg == nullptr )
    {
        *result_listener << "material array pointer is nullptr";
        return false;
    }
    std::vector<PartialMaterial> actualMaterials;
    actualMaterials.resize( index + 1 );
    OTK_ERROR_CHECK( cudaMemcpy( actualMaterials.data(), arg, sizeof( PartialMaterial ) * actualMaterials.size(), cudaMemcpyDeviceToHost ) );
    if( actualMaterials[index] != material )
    {
        *result_listener << "material " << index << " was " << actualMaterials[index] << ", expected " << material;
        return false;
    }

    *result_listener << "material " << index << " matches " << material;
    return true;
}

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

MATCHER_P2( hasDeviceTriangleUVPtr, index, uvs, "" )
{
    if( arg == nullptr )
    {
        *result_listener << "triangle normals pointer array is nullptr";
        return false;
    }
    std::vector<TriangleUVs*> actualPtrs;
    actualPtrs.resize( index + 1 );
    OTK_ERROR_CHECK( cudaMemcpy( actualPtrs.data(), arg, sizeof( TriangleUVs* ) * actualPtrs.size(), cudaMemcpyDeviceToHost ) );
    if( actualPtrs[index] == nullptr && uvs != nullptr )
    {
        *result_listener << "triangle normals pointer at index " << index << " is nullptr";
        return false;
    }
    if( actualPtrs[index] != uvs )
    {
        *result_listener << "triangle UVs pointer at index " << index << " is " << arg[index] << ", expected " << uvs;
        return false;
    }

    *result_listener << "triangle UVs pointer at index " << index << " matches " << uvs;
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

class MockDemandTexture : public StrictMock<demandLoading::DemandTexture>
{
  public:
    ~MockDemandTexture() override = default;

    MOCK_METHOD( unsigned, getId, (), ( const, override ) );
};

class MockDemandTextureCache : public StrictMock<DemandTextureCache>
{
  public:
    ~MockDemandTextureCache() override = default;

    MOCK_METHOD( uint_t, createDiffuseTextureFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( uint_t, createAlphaTextureFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( uint_t, createSkyboxTextureFromFile, ( const std::string& path ), ( override ) );
    MOCK_METHOD( DemandTextureCacheStatistics, getStatistics, (), ( const override ) );
};

class MockMaterialLoader : public StrictMock<demandMaterial::MaterialLoader>
{
  public:
    ~MockMaterialLoader() override = default;

    MOCK_METHOD( const char*, getCHFunctionName, (), ( const, override ) );
    MOCK_METHOD( uint_t, add, (), ( override ) );
    MOCK_METHOD( void, remove, ( uint_t ), ( override ) );
    MOCK_METHOD( std::vector<uint_t>, requestedMaterialIds, (), ( const, override ) );
    MOCK_METHOD( void, clearRequestedMaterialIds, (), ( override ) );
    MOCK_METHOD( bool, getRecycleProxyIds, (), ( const, override ) );
    MOCK_METHOD( void, setRecycleProxyIds, (bool), ( override ) );
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
};

using StrictMockDemandLoader    = StrictMock<MockDemandLoader>;
using StrictMockOptix           = StrictMock<MockOptix>;
using MockSceneLoaderPtr        = std::shared_ptr<MockSceneLoader>;
using MockDemandLoaderPtr       = std::shared_ptr<StrictMockDemandLoader>;
using MockDemandTextureCachePtr = std::shared_ptr<MockDemandTextureCache>;
using MockImageSourcePtr        = std::shared_ptr<MockImageSource>;
using MockMaterialLoaderPtr     = std::shared_ptr<MockMaterialLoader>;
using MockProxyFactoryPtr       = std::shared_ptr<MockProxyFactory>;
using MockRendererPtr           = std::shared_ptr<MockRenderer>;
using MockSceneProxyPtr         = std::shared_ptr<MockSceneProxy>;

using AccelBuildOptionsMatcher = Matcher<const OptixAccelBuildOptions*>;
using BuildInputMatcher        = Matcher<const OptixBuildInput*>;
using ProgramGroupDescMatcher  = Matcher<const OptixProgramGroupDesc*>;

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
    Expectation    expectModuleCreated( OptixModule module );
    ExpectationSet expectInitializeCreatesOptixState();
    ExpectationSet expectInitialize();

    CUstream                  m_stream{};
    StrictMockOptix           m_optix{};
    MockSceneLoaderPtr        m_sceneLoader{ std::make_shared<MockSceneLoader>() };
    MockDemandTextureCachePtr m_demandTextureCache{ std::make_shared<MockDemandTextureCache>() };
    MockProxyFactoryPtr       m_proxyFactory{ std::make_shared<MockProxyFactory>() };
    MockDemandLoaderPtr       m_demandLoader{ std::make_shared<StrictMockDemandLoader>() };
    MockGeometryLoaderPtr     m_geometryLoader{ std::make_shared<MockGeometryLoader>() };
    MockMaterialLoaderPtr     m_materialLoader{ std::make_shared<MockMaterialLoader>() };
    MockRendererPtr           m_renderer{ std::make_shared<MockRenderer>() };
    Options                   m_options{ testOptions() };
    // clang-format off
    PbrtScene m_scene{ m_options, m_sceneLoader, m_demandTextureCache, m_proxyFactory, m_demandLoader, m_geometryLoader, m_materialLoader, m_renderer };
    // clang-format on
    SceneDescriptionPtr            m_sceneDesc{ std::make_shared<otk::pbrt::SceneDescription>() };
    OptixAabb                      m_sceneBounds{ -1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f };
    MockSceneProxyPtr              m_mockSceneProxy{ std::make_shared<MockSceneProxy>() };
    uint_t                         m_scenePageId{ 6646U };
    demandGeometry::Context        m_demandGeomContext{ fakeDemandGeometryContext() };
    demandLoading::DeviceContext   m_demandLoadContext{ fakeDemandLoadingDeviceContext() };
    OptixTraversableHandle         m_fakeProxyTraversable{ 0xbaddf00dU };
    OptixTraversableHandle         m_fakeTopLevelTraversable{ 0xf01df01dU };
    OptixDeviceContext             m_fakeContext{ fakeOptixDeviceContext() };
    OptixAccelBufferSizes          m_topLevelASSizes{};
    OptixPipelineCompileOptions    m_pipelineCompileOptions{};
    OptixModule                    m_sceneModule{ reinterpret_cast<OptixModule>( 0x1111U ) };
    OptixModule                    m_builtinTriangleModule{ reinterpret_cast<OptixModule>( 0x4444U ) };
    OptixModule                    m_builtinSphereModule{ reinterpret_cast<OptixModule>( 0x5555U ) };
    std::vector<OptixProgramGroup> m_fakeProgramGroups{ PG( 111100U ), PG( 2222000U ), PG( 333300U ), PG( 444400U ),
                                                        PG( 555500U ), PG( 666600U ),  PG( 777700U ) };
    const char*                    m_proxyGeomIS{ "__intersection__proxyGeometry" };
    const char*                    m_proxyGeomCH{ "__closesthit__proxyGeometry" };
    const char*                    m_proxyMatCH{ "__closesthit__proxyMaterial" };
    const char*                    m_proxyMatMeshAlphaAH{ "__anyhit__alphaCutOutPartialMesh" };
    const char*                    m_proxyMatSphereAlphaAH{ "__anyhit__sphere" };
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

Expectation TestPbrtScene::expectModuleCreated( OptixModule module )
{
    Expectation expect;
#if OPTIX_VERSION < 70700
    expect = EXPECT_CALL( m_optix, moduleCreateFromPTX( m_fakeContext, _, _, _, _, _, _, _ ) )
                 .WillOnce( DoAll( SetArgPointee<7>( module ), Return( OPTIX_SUCCESS ) ) );
#else
    expect = EXPECT_CALL( m_optix, moduleCreate( m_fakeContext, _, _, _, _, _, _, _ ) )
                 .WillOnce( DoAll( SetArgPointee<7>( module ), Return( OPTIX_SUCCESS ) ) );
#endif
    return expect;
}

ExpectationSet TestPbrtScene::expectInitializeCreatesOptixState()
{
    ExpectationSet expect;
    expect += EXPECT_CALL( *m_sceneLoader, parseFile( _ ) ).WillOnce( Return( m_sceneDesc ) );
    expect += EXPECT_CALL( *m_mockSceneProxy, getPageId() ).WillOnce( Return( m_scenePageId ) );
    // TODO: Determine why adding this expectation to the set causes a dangling reference to m_mockSceneProxy
    /*expect +=*/EXPECT_CALL( *m_proxyFactory, scene( _, _ ) ).WillOnce( Return( m_mockSceneProxy ) );
    // This getter can be called anytime in any order.
    EXPECT_CALL( *m_renderer, getDeviceContext() ).WillRepeatedly( Return( m_fakeContext ) );
    expect += EXPECT_CALL( *m_geometryLoader, setSbtIndex( _ ) ).Times( AtLeast( 1 ) );
    expect += EXPECT_CALL( *m_geometryLoader, copyToDeviceAsync( m_stream ) ).Times( AtLeast( 1 ) );
    expect += EXPECT_CALL( *m_geometryLoader, createTraversable( _, _ ) ).WillOnce( Return( m_fakeProxyTraversable ) );
    // This getter can be called anytime in any order.
    EXPECT_CALL( *m_renderer, getPipelineCompileOptions() ).WillRepeatedly( ReturnRef( m_pipelineCompileOptions ) );
    expect += expectAccelComputeMemoryUsage( _, _ );
    expect += expectAccelBuild( _, _, m_fakeTopLevelTraversable );
    expect += expectModuleCreated( m_sceneModule );
    expect += EXPECT_CALL( m_optix, builtinISModuleGet( m_fakeContext, _, _, hasModuleTypeTriangle(), _ ) )
                  .WillOnce( DoAll( SetArgPointee<4>( m_builtinTriangleModule ), Return( OPTIX_SUCCESS ) ) );
    expect += EXPECT_CALL( m_optix, builtinISModuleGet( m_fakeContext, _, _, hasModuleTypeSphere(), _ ) )
                  .WillOnce( DoAll( SetArgPointee<4>( m_builtinSphereModule ), Return( OPTIX_SUCCESS ) ) );
    expect += EXPECT_CALL( *m_geometryLoader, getISFunctionName() ).WillRepeatedly( Return( m_proxyGeomIS ) );
    expect += EXPECT_CALL( *m_geometryLoader, getCHFunctionName() ).WillRepeatedly( Return( m_proxyGeomCH ) );
    expect += EXPECT_CALL( *m_materialLoader, getCHFunctionName() ).WillRepeatedly( Return( m_proxyMatCH ) );
    expect += EXPECT_CALL( m_optix, programGroupCreate( m_fakeContext, _, m_fakeProgramGroups.size(), _, _, _, _ ) )
                  .WillOnce( DoAll( SetArrayArgument<6>( m_fakeProgramGroups.begin(), m_fakeProgramGroups.end() ),
                                    Return( OPTIX_SUCCESS ) ) );
    expect += EXPECT_CALL( *m_renderer, setProgramGroups( _ ) ).Times( 1 );
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
    ExpectationSet expectProgramGroupAddedAfter( const ProgramGroupDescMatcher& desc, OptixProgramGroup result, const ExpectationSet& before );
    Expectation expectModuleCreatedAfter( OptixModule module, const ExpectationSet& before );
    Expectation expectRequestedProxyIdsAfter( std::initializer_list<uint_t> pageIds, const ExpectationSet& before );
    Expectation expectRequestedMaterialIdsAfter( std::initializer_list<uint_t> pageIds, const ExpectationSet& before );
    Expectation expectClearRequestedProxyIdsAfter( const ExpectationSet& before );
    Expectation expectClearRequestedMaterialIdsAfter( const ExpectationSet& before );
    Expectation expectGeometryLoaderGetContextAfter( const ExpectationSet& before );
    Expectation expectLaunchPrepareTrueAfter( const ExpectationSet& before );
    Expectation expectProxyDecomposableAfter( MockSceneProxyPtr proxy, bool decomposable, const ExpectationSet& before );
    Expectation expectProxyRemovedAfter( uint_t pageId, const ExpectationSet& before );
    Expectation expectSceneDecomposedAfterInitTo( const std::vector<SceneProxyPtr>& shapeProxies );
    Expectation expectGeometryLoaderCopiedToDeviceAfter( const ExpectationSet& first );
    Expectation expectGeometryLoaderCreateTraversableAfter( OptixTraversableHandle traversable, const ExpectationSet& before );
    Expectation expectProxyCreateGeometryAfter( MockSceneProxyPtr proxy, const GeometryInstance& geometry, const ExpectationSet& before );
    Expectation expectSceneProxyCreateGeometryAfter( const GeometryInstance& geometry, const ExpectationSet& before );
    Expectation expectMaterialLoaderAddReturnsAfter( uint_t materialId, const ExpectationSet& before );
    Expectation expectMaterialLoaderRemoveAfter( uint_t materialId, const ExpectationSet& before );
    GeometryInstance proxyMaterialTriMeshGeometry();
    GeometryInstance proxyMaterialSphereGeometry();
    ExpectationSet   expectNoGeometryResolvedAfter( const ExpectationSet& before );
    ExpectationSet   expectNoMaterialResolvedAfter( const ExpectationSet& before );
    ExpectationSet   expectNoTopLevelASBuildAfter( const ExpectationSet& before );
    void             setDistantLightOnSceneDescription();
    void             setInfiniteLightOnSceneDescription();

    std::vector<OptixProgramGroup> m_updatedGroups{ m_fakeProgramGroups };
    ExpectationSet                 m_init;
    OptixTraversableHandle         m_fakeUpdatedProxyTraversable{ 0xf00dbad2U };
    OptixTraversableHandle         m_fakeTriMeshTraversable{ 0xbeefbeefU };
    OptixTraversableHandle         m_fakeSphereTraversable{ 0x11110000U };
    OptixModule                    m_phongModule{ reinterpret_cast<OptixModule>( 0x3333U ) };
    OptixProgramGroup              m_fakePhongProgramGroup{ PG( 8888U ) };
    OptixProgramGroup              m_fakeAlphaPhongProgramGroup{ PG( 9999U ) };
    OptixProgramGroup              m_fakeDiffusePhongProgramGroup{ PG( 0x1010U ) };
    uint_t                         m_fakeMaterialId{ 666U };
    PhongMaterial                  m_realizedMaterial{
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

ExpectationSet TestPbrtSceneInitialized::expectProgramGroupAddedAfter( const ProgramGroupDescMatcher& desc,
                                                                       OptixProgramGroup              result,
                                                                       const ExpectationSet&          before )
{
    m_updatedGroups.push_back( result );
    ExpectationSet set;
    set += EXPECT_CALL( m_optix, programGroupCreate( m_fakeContext, desc, 1, _, _, _, _ ) )
               .After( before )
               .WillOnce( DoAll( SetArgPointee<6>( result ), Return( OPTIX_SUCCESS ) ) );
    set += EXPECT_CALL( *m_renderer, setProgramGroups( hasProgramGroupCount( m_updatedGroups.size() ) ) ).Times( 1 ).After( before );
    return set;
}

Expectation TestPbrtSceneInitialized::expectModuleCreatedAfter( OptixModule module, const ExpectationSet& before )
{
    Expectation expect;
#if OPTIX_VERSION < 70700
    expect = EXPECT_CALL( m_optix, moduleCreateFromPTX( m_fakeContext, _, _, _, _, _, _, _ ) )
                 .After( before )
                 .WillOnce( DoAll( SetArgPointee<7>( module ), Return( OPTIX_SUCCESS ) ) );
#else
    expect = EXPECT_CALL( m_optix, moduleCreate( m_fakeContext, _, _, _, _, _, _, _ ) )
                 .After( before )
                 .WillOnce( DoAll( SetArgPointee<7>( module ), Return( OPTIX_SUCCESS ) ) );
#endif
    return expect;
}

Expectation TestPbrtSceneInitialized::expectRequestedProxyIdsAfter( std::initializer_list<uint_t> pageIds, const ExpectationSet& before )
{
    return EXPECT_CALL( *m_geometryLoader, requestedProxyIds() ).After( before ).WillOnce( Return( std::vector<uint_t>{ pageIds } ) );
}

Expectation TestPbrtSceneInitialized::expectRequestedMaterialIdsAfter( std::initializer_list<uint_t> pageIds,
                                                                       const ExpectationSet&         before )
{
    return EXPECT_CALL( *m_materialLoader, requestedMaterialIds() ).After( before ).WillOnce( Return( std::vector<uint_t>{ pageIds } ) );
}

Expectation TestPbrtSceneInitialized::expectClearRequestedProxyIdsAfter( const ExpectationSet& before )
{
    return EXPECT_CALL( *m_geometryLoader, clearRequestedProxyIds() ).After( before );
}

Expectation TestPbrtSceneInitialized::expectClearRequestedMaterialIdsAfter( const ExpectationSet& before )
{
    return EXPECT_CALL( *m_materialLoader, clearRequestedMaterialIds() ).After( before );
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

Expectation TestPbrtSceneInitialized::expectMaterialLoaderAddReturnsAfter( uint_t materialId, const ExpectationSet& before )
{
    return EXPECT_CALL( *m_materialLoader, add() ).After( before ).WillOnce( Return( materialId ) );
}

Expectation TestPbrtSceneInitialized::expectMaterialLoaderRemoveAfter( uint_t materialId, const ExpectationSet& before )
{
    return EXPECT_CALL( *m_materialLoader, remove( materialId ) ).Times( 1 ).After( before );
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
    expect += EXPECT_CALL( *m_materialLoader, add() ).Times( 0 ).After( before );
    return expect;
}

ExpectationSet TestPbrtSceneInitialized::expectNoMaterialResolvedAfter( const ExpectationSet& before )
{
    ExpectationSet expect;
    expect += EXPECT_CALL( *m_materialLoader, requestedMaterialIds() ).Times( 0 ).After( before );
#if OPTIX_VERSION < 70700
    expect += EXPECT_CALL( m_optix, moduleCreateFromPTX( m_fakeContext, _, _, _, _, _, _, _ ) ).Times( 0 ).After( before );
#else
    expect += EXPECT_CALL( m_optix, moduleCreate( m_fakeContext, _, _, _, _, _, _, _ ) ).Times( 0 ).After( before );
#endif
    expect += EXPECT_CALL( m_optix, programGroupCreate( m_fakeContext, _, _, _, _, _, _ ) ).Times( 0 ).After( before );
    expect += EXPECT_CALL( *m_renderer, setProgramGroups( _ ) ).Times( 0 ).After( before );
    expect += EXPECT_CALL( *m_materialLoader, remove( m_fakeMaterialId ) ).Times( 0 ).After( before );
    return expect;
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
    EXPECT_CALL( *m_mockSceneProxy, getPageId() ).WillOnce( Return( m_scenePageId ) );
    EXPECT_CALL( *m_proxyFactory, scene( static_cast<GeometryLoaderPtr>( m_geometryLoader ), m_sceneDesc ) ).WillOnce( Return( m_mockSceneProxy ) );
    EXPECT_CALL( *m_renderer, getDeviceContext() ).WillRepeatedly( Return( m_fakeContext ) );
    EXPECT_CALL( *m_renderer, getPipelineCompileOptions() ).WillRepeatedly( ReturnRef( m_pipelineCompileOptions ) );
    EXPECT_CALL( *m_geometryLoader, setSbtIndex( _ ) ).Times( AtLeast( 1 ) );
    EXPECT_CALL( *m_geometryLoader, copyToDeviceAsync( m_stream ) ).Times( 1 );
    EXPECT_CALL( *m_geometryLoader, createTraversable( m_fakeContext, m_stream ) ).WillOnce( Return( m_fakeProxyTraversable ) );
    expectModuleCreated( m_sceneModule );
    EXPECT_CALL( m_optix, builtinISModuleGet( m_fakeContext, NotNull(), Pointee( Eq( m_pipelineCompileOptions ) ),
                                              AllOf( NotNull(), hasModuleTypeTriangle(), allowsRandomVertexAccess() ), NotNull() ) )
        .WillOnce( DoAll( SetArgPointee<4>( m_builtinTriangleModule ), Return( OPTIX_SUCCESS ) ) );
    EXPECT_CALL( m_optix, builtinISModuleGet( m_fakeContext, NotNull(), Pointee( Eq( m_pipelineCompileOptions ) ),
                                              AllOf( NotNull(), hasModuleTypeSphere(), allowsRandomVertexAccess() ), NotNull() ) )
        .WillOnce( DoAll( SetArgPointee<4>( m_builtinSphereModule ), Return( OPTIX_SUCCESS ) ) );
    EXPECT_CALL( *m_geometryLoader, getISFunctionName() ).WillRepeatedly( Return( m_proxyGeomIS ) );
    EXPECT_CALL( *m_geometryLoader, getCHFunctionName() ).WillRepeatedly( Return( m_proxyGeomCH ) );
    const char* const proxyMatIS = nullptr;
    EXPECT_CALL( *m_materialLoader, getCHFunctionName() ).WillRepeatedly( Return( m_proxyMatCH ) );
    size_t numGroups = m_fakeProgramGroups.size();
    auto   expectedProgramGroupDescs =
        AllOf( NotNull(), hasRayGenDesc( numGroups, m_sceneModule, "__raygen__perspectiveCamera" ),
               hasMissDesc( numGroups, m_sceneModule, "__miss__backgroundColor" ),
               hasHitGroupISCHDesc( numGroups, m_sceneModule, m_proxyGeomIS, m_sceneModule, m_proxyGeomCH ),
               hasHitGroupISCHDesc( numGroups, m_builtinTriangleModule, proxyMatIS, m_sceneModule, m_proxyMatCH ),
               hasHitGroupISAHCHDesc( numGroups, m_builtinTriangleModule, proxyMatIS, m_sceneModule,
                                      m_proxyMatMeshAlphaAH, m_sceneModule, m_proxyMatCH ),
               hasHitGroupISCHDesc( numGroups, m_builtinSphereModule, proxyMatIS, m_sceneModule, m_proxyMatCH ),
               hasHitGroupISAHCHDesc( numGroups, m_builtinSphereModule, proxyMatIS, m_sceneModule,
                                      m_proxyMatSphereAlphaAH, m_sceneModule, m_proxyMatCH ) );
    EXPECT_CALL( m_optix, programGroupCreate( m_fakeContext, expectedProgramGroupDescs, m_fakeProgramGroups.size(),
                                              NotNull(), NotNull(), NotNull(), NotNull() ) )
        .WillOnce( DoAll( SetArrayArgument<6>( m_fakeProgramGroups.begin(), m_fakeProgramGroups.end() ), Return( OPTIX_SUCCESS ) ) );
    EXPECT_CALL( *m_renderer, setProgramGroups( _ ) ).Times( 1 );
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
    expectRequestedMaterialIdsAfter( {}, m_init );

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
    expectRequestedMaterialIdsAfter( {}, m_init );

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
    expectRequestedMaterialIdsAfter( {}, m_init );

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
    expectRequestedMaterialIdsAfter( {}, m_init );
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
    for( OptixProgramGroup group : m_fakeProgramGroups )
    {
        EXPECT_CALL( m_optix, programGroupDestroy( group ) ).After( m_init ).WillOnce( Return( OPTIX_SUCCESS ) );
    }
    EXPECT_CALL( m_optix, moduleDestroy( m_sceneModule ) ).After( m_init ).WillOnce( Return( OPTIX_SUCCESS ) );

    m_scene.cleanup();
}

TEST_F( TestPbrtSceneInitialized, firstLaunchResolvesSceneProxyToSingleTriMeshWithProxyMaterial )
{
    expectGeometryLoaderGetContextAfter( m_init );
    expectRequestedProxyIdsAfter( { m_scenePageId }, m_init );
    expectClearRequestedProxyIdsAfter( m_init );
    expectProxyDecomposableAfter( m_mockSceneProxy, false, m_init );
    expectProxyRemovedAfter( m_scenePageId, m_init );
    OptixTraversableHandle updatedProxyTravHandle{ 0xf00d13f00d13U };
    expectGeometryLoaderCopiedToDeviceAfter( m_init );
    expectGeometryLoaderCreateTraversableAfter( updatedProxyTravHandle, m_init );
    const GeometryInstance proxyMaterialGeom = proxyMaterialTriMeshGeometry();
    expectSceneProxyCreateGeometryAfter( proxyMaterialGeom, m_init );
    expectMaterialLoaderAddReturnsAfter( m_fakeMaterialId, m_init );
    expectRequestedMaterialIdsAfter( {}, m_init );
    const auto isTopLevelIAS = twoInstanceIAS( updatedProxyTravHandle, proxyMaterialGeom.instance.traversableHandle,
                                               +HitGroupIndex::PROXY_MATERIAL_TRIANGLE, m_fakeMaterialId );
    expectCreateTopLevelTraversable( isTopLevelIAS, m_fakeTopLevelTraversable, m_init );
    expectLaunchPrepareTrueAfter( m_init );

    Params      params{};
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    EXPECT_EQ( 1, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 1, stats.numGeometriesRealized );
    EXPECT_EQ( 1, stats.numProxyMaterialsCreated );
    EXPECT_EQ( 0, stats.numPartialMaterialsRealized );
    EXPECT_EQ( 0, stats.numPartialMaterialsRealized );
    EXPECT_EQ( 0, stats.numMaterialsRealized );
}

TEST_F( TestPbrtSceneInitialized, firstLaunchResolvesNoGeometryUntilOneShotFired )
{
    m_options.oneShotGeometry = true;
    expectGeometryLoaderGetContextAfter( m_init );
    expectNoGeometryResolvedAfter( m_init );
    expectRequestedMaterialIdsAfter( {}, m_init );
    expectNoTopLevelASBuildAfter( m_init );
    expectLaunchPrepareTrueAfter( m_init );

    Params      params{};
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    EXPECT_EQ( 0, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 0, stats.numGeometriesRealized );
    EXPECT_EQ( 0, stats.numProxyMaterialsCreated );
    EXPECT_EQ( 0, stats.numPartialMaterialsRealized );
    EXPECT_EQ( 0, stats.numMaterialsRealized );
}

TEST_F( TestPbrtSceneInitialized, resolveOneGeometryAfterOneShotFired )
{
    m_options.oneShotGeometry = true;
    ExpectationSet first;
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
    first += expectMaterialLoaderAddReturnsAfter( m_fakeMaterialId, m_init );
    first += expectRequestedMaterialIdsAfter( {}, m_init );
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
    expectRequestedMaterialIdsAfter( {}, first );
    expectNoTopLevelASBuildAfter( first );
    expectLaunchPrepareTrueAfter( first );

    Params params{};
    m_scene.resolveOneGeometry();
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    EXPECT_EQ( 1, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 1, stats.numGeometriesRealized );
    EXPECT_EQ( 1, stats.numProxyMaterialsCreated );
    EXPECT_EQ( 0, stats.numPartialMaterialsRealized );
    EXPECT_EQ( 0, stats.numMaterialsRealized );
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
    expectRequestedMaterialIdsAfter( {}, m_init );
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
    EXPECT_EQ( 0, stats.numProxyMaterialsCreated );
    EXPECT_EQ( 0, stats.numPartialMaterialsRealized );
    EXPECT_EQ( 0, stats.numMaterialsRealized );
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
    first += expectRequestedMaterialIdsAfter( {}, m_init );
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
    expectRequestedMaterialIdsAfter( {}, first );
    expectProxyDecomposableAfter( sphereProxy, false, first );
    expectProxyDecomposableAfter( triMeshProxy, false, first );
    expectProxyRemovedAfter( spherePageId, first );
    expectProxyRemovedAfter( triMeshPageId, first );
    ExpectationSet createSphere = expectProxyCreateGeometryAfter( sphereProxy, proxyMaterialSphereGeometry(), first );
    const uint_t   sphereMaterialId{ 3333 };
    expectMaterialLoaderAddReturnsAfter( sphereMaterialId, createSphere );
    ExpectationSet createTriMesh = expectProxyCreateGeometryAfter( triMeshProxy, proxyMaterialTriMeshGeometry(), first );
    const uint_t triMeshMaterialId{ 4444 };
    expectMaterialLoaderAddReturnsAfter( triMeshMaterialId, createTriMesh );
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
    EXPECT_EQ( 2, stats.numProxyMaterialsCreated );
    EXPECT_EQ( 0, stats.numPartialMaterialsRealized );
    EXPECT_EQ( 0, stats.numMaterialsRealized );
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
    m_first += expectProxyDecomposableAfter( m_mockSceneProxy, false, m_init );
    m_first += expectProxyRemovedAfter( m_scenePageId, m_init );
    m_first += expectGeometryLoaderCopiedToDeviceAfter( m_init );
    m_first += expectGeometryLoaderCreateTraversableAfter( m_fakeUpdatedProxyTraversable, m_init );
    m_first += expectSceneProxyCreateGeometryAfter( sceneGeometry(), m_init );
    m_first += expectMaterialLoaderAddReturnsAfter( m_fakeMaterialId, m_init );
    appendTo( m_first, expectCreateTopLevelTraversable( _, m_fakeTopLevelTraversable, m_init ) );
    m_first += expectLaunchPrepareTrueAfter( m_init );
}

class TestPbrtSceneSecondLaunch : public TestPbrtSceneFirstLaunch
{
  public:
    ~TestPbrtSceneSecondLaunch() override = default;

  protected:
    ExpectationSet expectSecondLaunchCreatesHitGroupAndAccel( const ProgramGroupDescMatcher& hitGroupDesc,
                                                              const BuildInputMatcher&       buildInput )
    {
        ExpectationSet second;
        second += expectRequestedProxyIdsAfter( {}, m_first );
        second += expectClearRequestedProxyIdsAfter( m_first );
        second += expectRequestedMaterialIdsAfter( { m_fakeMaterialId }, m_first );
        second += expectClearRequestedMaterialIdsAfter( m_first );
        second += expectModuleCreatedAfter( m_phongModule, m_first );
        appendTo( second, expectProgramGroupAddedAfter( hitGroupDesc, m_fakePhongProgramGroup, m_first ) );
        second += expectMaterialLoaderRemoveAfter( m_fakeMaterialId, m_first );
        appendTo( second, expectCreateTopLevelTraversable( buildInput, m_fakeTopLevelTraversable, m_first ) );
        second += expectGeometryLoaderGetContextAfter( m_first );
        second += expectLaunchPrepareTrueAfter( m_first );
        return second;
    }

    std::vector<OptixProgramGroup> m_expectedGroups{};
};

class TestPbrtSceneSphere : public TestPbrtSceneSecondLaunch
{
  public:
    ~TestPbrtSceneSphere() override = default;

  protected:
    GeometryInstance sceneGeometry() override { return proxyMaterialSphereGeometry(); }
};

}  // namespace

TEST_F( TestPbrtSceneSphere, resolveProxyMaterialToPhongMaterial )
{
    m_first += expectRequestedMaterialIdsAfter( {}, m_init );
    const char* const sphereIS = nullptr;
    const char* const sphereCH = "__closesthit__sphere";
    const auto        expectedHitGroupDesc =
        AllOf( NotNull(), hasHitGroupISCHDesc( 1, m_builtinSphereModule, sphereIS, m_phongModule, sphereCH ) );
    const auto isIAS =
        twoInstanceIAS( m_fakeUpdatedProxyTraversable, m_fakeSphereTraversable, +HitGroupIndex::REALIZED_MATERIAL_START );
    expectSecondLaunchCreatesHitGroupAndAccel( expectedHitGroupDesc, isIAS );

    Params params{};
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    const bool launchNeeded = m_scene.beforeLaunch( m_stream, params );

    EXPECT_TRUE( launchNeeded );
    EXPECT_THAT( params.realizedMaterials, hasDeviceMaterial( 0, m_realizedMaterial ) );
    EXPECT_NE( nullptr, params.instanceNormals );
    EXPECT_THAT( params.instanceNormals, hasDeviceTriangleNormalPtr( 0, null_cast<TriangleNormals>() ) );
}

namespace {

class TestPbrtSceneTriMesh : public TestPbrtSceneSecondLaunch
{
  public:
    ~TestPbrtSceneTriMesh() override = default;

  protected:
    GeometryInstance sceneGeometry() override { return proxyMaterialTriMeshGeometry(); }
};

}  // namespace

TEST_F( TestPbrtSceneTriMesh, resolveProxyMaterialToPhongMaterial )
{
    m_first += expectRequestedMaterialIdsAfter( {}, m_init );
    const char* const phongMatIS = nullptr;
    const char* const phongMatCH = "__closesthit__mesh";
    const auto        expectedHitGroupDesc =
        AllOf( NotNull(), hasHitGroupISCHDesc( 1, m_builtinTriangleModule, phongMatIS, m_phongModule, phongMatCH ) );
    const auto isIAS =
        twoInstanceIAS( m_fakeUpdatedProxyTraversable, m_fakeTriMeshTraversable, +HitGroupIndex::REALIZED_MATERIAL_START );
    expectSecondLaunchCreatesHitGroupAndAccel( expectedHitGroupDesc, isIAS );

    Params params{};
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    EXPECT_THAT( params.realizedMaterials, hasDeviceMaterial( 0, m_realizedMaterial ) );
    EXPECT_THAT( params.instanceNormals, hasDeviceTriangleNormalPtr( 0, m_fakeTriangleNormals ) );
    EXPECT_EQ( 1, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 1, stats.numGeometriesRealized );
    EXPECT_EQ( 1, stats.numProxyMaterialsCreated );
    EXPECT_EQ( 0, stats.numPartialMaterialsRealized );
    EXPECT_EQ( 1, stats.numMaterialsRealized );
}

TEST_F( TestPbrtSceneTriMesh, resolvesNoMaterialsUntilOneShotFired )
{
    m_options.oneShotMaterial = true;
    appendTo( m_first, expectNoMaterialResolvedAfter( m_init ) );
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
    EXPECT_EQ( 1, stats.numProxyMaterialsCreated );
    EXPECT_EQ( 0, stats.numPartialMaterialsRealized );
    EXPECT_EQ( 0, stats.numMaterialsRealized );
}

TEST_F( TestPbrtSceneTriMesh, resolveOneMaterialAfterOneShotFired )
{
    m_options.oneShotMaterial = true;
    m_first += expectRequestedMaterialIdsAfter( {}, m_init );
    const char* const triMeshIS = nullptr;
    const char* const triMeshCH = "__closesthit__mesh";
    const auto        expectedHitGroupDesc =
        AllOf( NotNull(), hasHitGroupISCHDesc( 1, m_builtinTriangleModule, triMeshIS, m_phongModule, triMeshCH ) );
    const auto isIAS =
        twoInstanceIAS( m_fakeUpdatedProxyTraversable, m_fakeTriMeshTraversable, +HitGroupIndex::REALIZED_MATERIAL_START );
    ExpectationSet second = expectSecondLaunchCreatesHitGroupAndAccel( expectedHitGroupDesc, isIAS );
    expectRequestedProxyIdsAfter( {}, second );
    expectClearRequestedProxyIdsAfter( second );
    expectGeometryLoaderGetContextAfter( second );
    expectLaunchPrepareTrueAfter( second );

    Params params{};
    m_scene.resolveOneMaterial();
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    EXPECT_THAT( params.realizedMaterials, hasDeviceMaterial( 0, m_realizedMaterial ) );
    EXPECT_THAT( params.instanceNormals, hasDeviceTriangleNormalPtr( 0, m_fakeTriangleNormals ) );
    EXPECT_EQ( 1, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 1, stats.numGeometriesRealized );
    EXPECT_EQ( 1, stats.numProxyMaterialsCreated );
    EXPECT_EQ( 0, stats.numPartialMaterialsRealized );
    EXPECT_EQ( 1, stats.numMaterialsRealized );
}

namespace {

class TestPbrtSceneAlphaMapTriMesh : public TestPbrtSceneTriMesh
{
  public:
    ~TestPbrtSceneAlphaMapTriMesh() override = default;

  protected:
    GeometryInstance sceneGeometry() override;
    ExpectationSet   expectSecondLaunchCreatesAlphaMapForId( uint_t textureId );

    MockImageSourcePtr m_imageSource{ std::make_shared<MockImageSource>() };
    MockDemandTexture  m_texture;
};

GeometryInstance TestPbrtSceneAlphaMapTriMesh::sceneGeometry()
{
    GeometryInstance geometry = proxyMaterialTriMeshGeometry();
    geometry.material.flags   = MaterialFlags::ALPHA_MAP;
    geometry.alphaMapFileName = "alphaMap.png";
    return geometry;
}

ExpectationSet TestPbrtSceneAlphaMapTriMesh::expectSecondLaunchCreatesAlphaMapForId( uint_t textureId )
{
    ExpectationSet second;
    second += expectRequestedProxyIdsAfter( {}, m_first );
    second += expectClearRequestedProxyIdsAfter( m_first );
    second += expectRequestedMaterialIdsAfter( { m_fakeMaterialId }, m_first );
    second +=
        EXPECT_CALL( *m_demandTextureCache, createAlphaTextureFromFile( StrEq( "alphaMap.png" ) ) ).After( m_first ).WillOnce( Return( textureId ) );
    second += expectGeometryLoaderGetContextAfter( m_first );
    second += expectLaunchPrepareTrueAfter( m_first );
    const auto isAlphaMapIAS = twoInstanceIAS( m_fakeUpdatedProxyTraversable, m_fakeTriMeshTraversable,
                                               +HitGroupIndex::PROXY_MATERIAL_TRIANGLE_ALPHA, m_fakeMaterialId );
    appendTo( second, expectCreateTopLevelTraversable( isAlphaMapIAS, m_fakeTopLevelTraversable, m_first ) );
    return second;
}

}  // namespace

TEST_F( TestPbrtSceneAlphaMapTriMesh, resolveProxyMaterialToAlphaMapProxyMaterial )
{
    // first launch: resolve geometry
    m_first += expectRequestedMaterialIdsAfter( {}, m_init );
    // second launch: resolve alpha material phase 1
    const uint_t textureId{ 8686U };
    expectSecondLaunchCreatesAlphaMapForId( textureId );

    Params params{};
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    const PartialMaterial expected{ textureId };
    EXPECT_THAT( params.partialMaterials, hasDevicePartialMaterial( m_fakeMaterialId, expected ) );
    EXPECT_THAT( params.partialUVs, hasDeviceTriangleUVPtr( m_fakeMaterialId, m_fakeTriangleUVs ) );
    EXPECT_EQ( nullptr, params.realizedMaterials );
    EXPECT_EQ( nullptr, params.instanceNormals );
    EXPECT_EQ( nullptr, params.instanceUVs );
    EXPECT_EQ( 1, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 1, stats.numGeometriesRealized );
    EXPECT_EQ( 1, stats.numProxyMaterialsCreated );
    EXPECT_EQ( 1, stats.numPartialMaterialsRealized );
    EXPECT_EQ( 0, stats.numMaterialsRealized );
}

TEST_F( TestPbrtSceneAlphaMapTriMesh, resolveProxyAlphaMapMaterialToPhongMaterial )
{
    // first launch: resolve geometry
    m_first += expectRequestedMaterialIdsAfter( {}, m_init );
    // second launch: resolve alpha material phase 1
    const uint_t   textureId{ 8686U };
    ExpectationSet second = expectSecondLaunchCreatesAlphaMapForId( textureId );
    // third launch: resolve alpha material phase 2
    expectRequestedProxyIdsAfter( {}, second );
    expectClearRequestedProxyIdsAfter( second );
    expectRequestedMaterialIdsAfter( { m_fakeMaterialId }, second );
    expectClearRequestedMaterialIdsAfter( second );
    expectModuleCreatedAfter( m_phongModule, second );
    const char* const triMeshIS = nullptr;
    const char* const triMeshAH = "__anyhit__alphaCutOutMesh";
    const char* const triMeshCH = "__closesthit__mesh";
    auto expectedHitGroupDesc = AllOf( NotNull(), hasHitGroupISAHCHDesc( 1, m_builtinTriangleModule, triMeshIS, m_sceneModule,
                                                                         triMeshAH, m_phongModule, triMeshCH ) );
    expectProgramGroupAddedAfter( expectedHitGroupDesc, m_fakeAlphaPhongProgramGroup, second );
    expectMaterialLoaderRemoveAfter( m_fakeMaterialId, second );
    expectCreateTopLevelTraversable( twoInstanceIAS( m_fakeUpdatedProxyTraversable, m_fakeTriMeshTraversable,
                                                     +HitGroupIndex::REALIZED_MATERIAL_START ),
                                     m_fakeTopLevelTraversable, second );
    expectGeometryLoaderGetContextAfter( second );
    expectLaunchPrepareTrueAfter( second );

    Params params{};
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    const uint_t  realizedGeometryInstanceIndex{};
    PhongMaterial expectedRealizedMaterial{ m_realizedMaterial };
    expectedRealizedMaterial.flags          = MaterialFlags::ALPHA_MAP | MaterialFlags::ALPHA_MAP_ALLOCATED;
    expectedRealizedMaterial.alphaTextureId = textureId;
    EXPECT_THAT( params.realizedMaterials, hasDeviceMaterial( realizedGeometryInstanceIndex, expectedRealizedMaterial ) );
    EXPECT_THAT( params.instanceUVs, hasDeviceTriangleUVPtr( realizedGeometryInstanceIndex, m_fakeTriangleUVs ) );
    const PartialMaterial expected{ 0 };
    EXPECT_THAT( params.partialMaterials, hasDevicePartialMaterial( m_fakeMaterialId, expected ) );
    EXPECT_THAT( params.partialUVs, hasDeviceTriangleUVPtr( m_fakeMaterialId, null_cast<TriangleUVs>() ) );
    EXPECT_EQ( 1, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 1, stats.numGeometriesRealized );
    EXPECT_EQ( 1, stats.numProxyMaterialsCreated );
    EXPECT_EQ( 1, stats.numPartialMaterialsRealized );
    EXPECT_EQ( 1, stats.numMaterialsRealized );
}

namespace {

class TestPbrtSceneDiffuseMapTriMesh : public TestPbrtSceneTriMesh
{
  public:
    ~TestPbrtSceneDiffuseMapTriMesh() override = default;

  protected:
    GeometryInstance sceneGeometry() override;
    ExpectationSet   expectSecondLaunchCreatesDiffuseMapForId( uint_t textureId, ExpectationSet& second );

    MockImageSourcePtr m_imageSource{ std::make_shared<MockImageSource>() };
    MockDemandTexture  m_texture;
};

GeometryInstance TestPbrtSceneDiffuseMapTriMesh::sceneGeometry()
{
    GeometryInstance geometry   = proxyMaterialTriMeshGeometry();
    geometry.material.flags     = MaterialFlags::DIFFUSE_MAP;
    geometry.diffuseMapFileName = "diffuseMap.png";
    return geometry;
}

ExpectationSet TestPbrtSceneDiffuseMapTriMesh::expectSecondLaunchCreatesDiffuseMapForId( uint_t textureId, ExpectationSet& second )
{
    second +=
        EXPECT_CALL( *m_demandTextureCache, createDiffuseTextureFromFile( StrEq( "diffuseMap.png" ) ) ).After( m_first ).WillOnce( Return( textureId ) );
    return second;
}

}  // namespace

TEST_F( TestPbrtSceneDiffuseMapTriMesh, resolveProxyMaterialToDiffuseMapMaterial )
{
    // first launch: resolve geometry
    m_first += expectRequestedMaterialIdsAfter( {}, m_init );
    // second launch: resolve diffuse material
    const char* const triMeshIS = nullptr;
    const char* const triMeshCH = "__closesthit__texturedMesh";
    const auto        expectedHitGroupDesc =
        AllOf( NotNull(), hasHitGroupISCHDesc( 1, m_builtinTriangleModule, triMeshIS, m_sceneModule, triMeshCH ) );
    const auto isPhongIAS =
        twoInstanceIAS( m_fakeUpdatedProxyTraversable, m_fakeTriMeshTraversable, +HitGroupIndex::REALIZED_MATERIAL_START );
    ExpectationSet second = expectSecondLaunchCreatesHitGroupAndAccel( expectedHitGroupDesc, isPhongIAS );
    const uint_t   textureId{ 8686U };
    expectSecondLaunchCreatesDiffuseMapForId( textureId, second );

    Params params{};
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    EXPECT_EQ( nullptr, params.partialMaterials );
    EXPECT_EQ( nullptr, params.partialUVs );
    EXPECT_NE( nullptr, params.realizedMaterials );
    EXPECT_NE( nullptr, params.instanceNormals );
    EXPECT_NE( nullptr, params.instanceUVs );
    const uint_t  realizedGeometryInstanceIndex{};
    PhongMaterial expectedTexturedMaterial{ m_realizedMaterial };
    expectedTexturedMaterial.flags            = MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED;
    expectedTexturedMaterial.diffuseTextureId = textureId;
    EXPECT_THAT( params.realizedMaterials, hasDeviceMaterial( realizedGeometryInstanceIndex, expectedTexturedMaterial ) );
    EXPECT_THAT( params.instanceUVs, hasDeviceTriangleUVPtr( realizedGeometryInstanceIndex, m_fakeTriangleUVs ) );
    EXPECT_EQ( 1, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 1, stats.numGeometriesRealized );
    EXPECT_EQ( 1, stats.numProxyMaterialsCreated );
    EXPECT_EQ( 0, stats.numPartialMaterialsRealized );
    EXPECT_EQ( 1, stats.numMaterialsRealized );
}


namespace {

class TestPbrtSceneDiffuseAlphaMapTriMesh : public TestPbrtSceneTriMesh
{
  public:
    ~TestPbrtSceneDiffuseAlphaMapTriMesh() override = default;

  protected:
    GeometryInstance sceneGeometry() override;
    void             expectSecondLaunchCreatesAlphaTextureAndAccel( uint_t textureId );
    void             expectThirdLaunchCreatesDiffuseTextureProgramGroupAndAccel( uint_t textureId );

    ExpectationSet m_second;
};

GeometryInstance TestPbrtSceneDiffuseAlphaMapTriMesh::sceneGeometry()
{
    GeometryInstance geometry   = proxyMaterialTriMeshGeometry();
    geometry.material.flags     = MaterialFlags::DIFFUSE_MAP | MaterialFlags::ALPHA_MAP;
    geometry.diffuseMapFileName = "diffuseMap.png";
    geometry.alphaMapFileName   = "alphaMap.png";
    return geometry;
}

void TestPbrtSceneDiffuseAlphaMapTriMesh::expectSecondLaunchCreatesAlphaTextureAndAccel( uint_t textureId )
{
    m_second += expectRequestedProxyIdsAfter( {}, m_first );
    m_second += expectClearRequestedProxyIdsAfter( m_first );
    m_second += expectRequestedMaterialIdsAfter( { m_fakeMaterialId }, m_first );
    m_second +=
        EXPECT_CALL( *m_demandTextureCache, createAlphaTextureFromFile( StrEq( "alphaMap.png" ) ) ).After( m_first ).WillOnce( Return( textureId ) );
    m_second += expectGeometryLoaderGetContextAfter( m_first );
    m_second += expectLaunchPrepareTrueAfter( m_first );
    const auto isAlphaMapIAS =
        AllOf( NotNull(),
               hasInstanceBuildInput(
                   0, hasAll( hasNumInstances( 2 ),
                              hasDeviceInstances( hasInstance( 0, hasInstanceTraversable( m_fakeUpdatedProxyTraversable ) ),
                                                  hasInstance( 1, hasInstanceTraversable( m_fakeTriMeshTraversable ),
                                                               hasInstanceSbtOffset( +HitGroupIndex::PROXY_MATERIAL_TRIANGLE_ALPHA ),
                                                               hasInstanceId( m_fakeMaterialId ) ) ) ) ) );
    appendTo( m_second, expectCreateTopLevelTraversable( isAlphaMapIAS, m_fakeTopLevelTraversable, m_first ) );
}

void TestPbrtSceneDiffuseAlphaMapTriMesh::expectThirdLaunchCreatesDiffuseTextureProgramGroupAndAccel( uint_t textureId )
{
    expectRequestedProxyIdsAfter( {}, m_second );
    expectClearRequestedProxyIdsAfter( m_second );
    expectRequestedMaterialIdsAfter( { m_fakeMaterialId }, m_second );
    expectClearRequestedMaterialIdsAfter( m_second );
    EXPECT_CALL( *m_demandTextureCache, createDiffuseTextureFromFile( StrEq( "diffuseMap.png" ) ) ).After( m_second ).WillOnce( Return( textureId ) );

    expectModuleCreatedAfter( m_phongModule, m_second );
    const char* const triMeshIS = nullptr;
    const char* const triMeshAH = "__anyhit__alphaCutOutMesh";
    const char* const triMeshCH = "__closesthit__texturedMesh";
    const auto expectedHitGroupDesc = AllOf( NotNull(), hasHitGroupISAHCHDesc( 1, m_builtinTriangleModule, triMeshIS, m_sceneModule,
                                                                               triMeshAH, m_sceneModule, triMeshCH ) );
    expectProgramGroupAddedAfter( expectedHitGroupDesc, m_fakeAlphaPhongProgramGroup, m_second );
    expectMaterialLoaderRemoveAfter( m_fakeMaterialId, m_second );
    const auto isPhongIAS =
        AllOf( NotNull(),
               hasInstanceBuildInput(
                   0, hasAll( hasNumInstances( 2 ),
                              hasDeviceInstances( hasInstance( 0, hasInstanceTraversable( m_fakeUpdatedProxyTraversable ) ),
                                                  hasInstance( 1, hasInstanceTraversable( m_fakeTriMeshTraversable ),
                                                               hasInstanceSbtOffset( +HitGroupIndex::REALIZED_MATERIAL_START ),
                                                               hasInstanceId( 0 ) ) ) ) ) );
    expectCreateTopLevelTraversable( isPhongIAS, m_fakeTopLevelTraversable, m_second );
    expectGeometryLoaderGetContextAfter( m_second );
    expectLaunchPrepareTrueAfter( m_second );
}

}  // namespace

TEST_F( TestPbrtSceneDiffuseAlphaMapTriMesh, resolveProxyMaterialToDiffuseAlphaMapMaterial )
{
    // first launch: resolve geometry
    m_first += expectRequestedMaterialIdsAfter( {}, m_init );
    // second launch: resolve alpha material phase 1
    const uint_t alphaTextureId{ 8686U };
    expectSecondLaunchCreatesAlphaTextureAndAccel( alphaTextureId );
    // third launch: resolve alpha material phase 2
    const uint_t diffuseTextureId{ 2323U };
    expectThirdLaunchCreatesDiffuseTextureProgramGroupAndAccel( diffuseTextureId );

    Params params{};
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    EXPECT_TRUE( m_scene.beforeLaunch( m_stream, params ) );
    const bool  launchNeeded = m_scene.beforeLaunch( m_stream, params );
    const Stats stats        = m_scene.getStatistics();

    EXPECT_TRUE( launchNeeded );
    const uint_t  realizedGeometryInstanceIndex{};
    PhongMaterial expectedRealizedMaterial{ m_realizedMaterial };
    expectedRealizedMaterial.flags = MaterialFlags::ALPHA_MAP | MaterialFlags::ALPHA_MAP_ALLOCATED
                                     | MaterialFlags::DIFFUSE_MAP | MaterialFlags::DIFFUSE_MAP_ALLOCATED;
    expectedRealizedMaterial.alphaTextureId   = alphaTextureId;
    expectedRealizedMaterial.diffuseTextureId = diffuseTextureId;
    EXPECT_THAT( params.realizedMaterials, hasDeviceMaterial( realizedGeometryInstanceIndex, expectedRealizedMaterial ) );
    EXPECT_THAT( params.instanceUVs, hasDeviceTriangleUVPtr( realizedGeometryInstanceIndex, m_fakeTriangleUVs ) );
    const PartialMaterial expected{ 0 };
    EXPECT_THAT( params.partialMaterials, hasDevicePartialMaterial( m_fakeMaterialId, expected ) );
    EXPECT_THAT( params.partialUVs, hasDeviceTriangleUVPtr( m_fakeMaterialId, null_cast<TriangleUVs>() ) );
    EXPECT_EQ( 1, stats.numProxyGeometriesResolved );
    EXPECT_EQ( 1, stats.numGeometriesRealized );
    EXPECT_EQ( 1, stats.numProxyMaterialsCreated );
    EXPECT_EQ( 1, stats.numPartialMaterialsRealized );
    EXPECT_EQ( 1, stats.numMaterialsRealized );
}
