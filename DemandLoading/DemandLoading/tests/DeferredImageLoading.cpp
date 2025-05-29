// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DeferredImageLoadingKernels.h"
#include "TestDemandLoadingKernelsCuda.h"

#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>
#include <OptiXToolkit/OptiXMemory/CompileOptions.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

#if OPTIX_VERSION < 70700
#define optixModuleCreate optixModuleCreateFromPTX
#endif

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenSbtRecord = SbtRecord<RayGenData>;
using MissSbtRecord   = SbtRecord<int>;
using uint_t          = unsigned int;

static void contextLog( uint_t level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 )
              << tag << "]: " << message << '\n';
}

static const int NUM_PAYLOAD_VALUES   = 0;
static const int NUM_ATTRIBUTE_VALUES = 0;
static const int OUTPUT_WIDTH         = 1;
static const int OUTPUT_HEIGHT        = 1;

namespace {

class MockImageSource : public imageSource::ImageSource
{
  public:
    ~MockImageSource() override = default;

    MOCK_METHOD( void, open, ( imageSource::TextureInfo * info ), ( override ) );
    MOCK_METHOD( void, close, (), ( override ) );
    MOCK_METHOD( bool, isOpen, (), ( const override ) );
    MOCK_METHOD( const imageSource::TextureInfo&, getInfo, (), ( const override ) );
    MOCK_METHOD( CUmemorytype, getFillType, (), ( const override ) );
    MOCK_METHOD( bool, readTile, ( char* dest, unsigned int mipLevel, const imageSource::Tile& tile, CUstream stream ), ( override ) );
    MOCK_METHOD( bool,
                 readMipLevel,
                 ( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream ),
                 ( override ) );
    MOCK_METHOD( bool,
                 readMipTail,
                 ( char* dest, unsigned int mipTailFirstLevel, unsigned int numMipLevels, const uint2* mipLevelDims, CUstream stream ),
                 ( override ) );
    MOCK_METHOD( bool, readBaseColor, ( float4 & dest ), ( override ) );
    MOCK_METHOD( unsigned int, getTileWidth, (), ( const override ) );
    MOCK_METHOD( unsigned int, getTileHeight, (), ( const override ) );
    MOCK_METHOD( unsigned long long, getNumTilesRead, (), ( const override ) );
    MOCK_METHOD( unsigned long long, getNumBytesRead, (), ( const override ) );
    MOCK_METHOD( double, getTotalReadTime, (), ( const override ) );
    MOCK_METHOD( bool, hasCascade, (), ( const override ) );
};

class DeferredImageLoadingTest : public testing::Test
{
  public:
    DeferredImageLoadingTest()
    {
        cudaFree( nullptr );
        initDemandLoading();
        createContext();
        initPipelineOpts();
        createModules();
        createProgramGroups();
        createPipeline();
        buildShaderBindingTable();
        allocateParams();
        allocateOutput();
    }

    ~DeferredImageLoadingTest()
    {
        freeOutput();
        freeParams();
        freeShaderBindingTable();
        freePipeline();
        freeProgramGroups();
        freeModules();
        freeContext();
        freeDemandLoading();
    }

protected:
    enum
    {
        GROUP_RAYGEN = 0,
        GROUP_MISS,
        NUM_GROUPS
    };

    void initDemandLoading();
    void createContext();
    void initPipelineOpts();
    void createModules();
    void createProgramGroups();
    void createPipeline();
    void buildShaderBindingTable();
    void allocateParams();
    void allocateOutput();
    void freeOutput();
    void freeParams();
    void freeShaderBindingTable();
    void freePipeline();
    void freeProgramGroups();
    void freeModules();
    void freeContext();
    void freeDemandLoading();

    void launchAndWaitForRequests();

    demandLoading::DemandLoader* m_loader;
    uint_t                       m_deviceIndex;
    CUcontext                    m_cudaContext;
    CUstream                     m_stream;
    OptixDeviceContext           m_context;
    OptixPipelineCompileOptions  m_pipelineOpts{};
    OptixModule                  m_module{};
    OptixProgramGroup            m_groups[NUM_GROUPS]{};
    OptixPipeline                m_pipeline{};
    OptixShaderBindingTable      m_sbt{};
    Params                       m_params{};
    void*                        m_devParams{};
    float4*                      m_devOutput{};
};

void DeferredImageLoadingTest::initDemandLoading()
{
    demandLoading::Options options{};
#ifndef NDEBUG
    options.maxThreads = 1;
#endif

    m_deviceIndex = demandLoading::getFirstSparseTextureDevice();
    if( m_deviceIndex == demandLoading::MAX_DEVICES )
        throw std::runtime_error( "No devices support demand loading" );

    OTK_ERROR_CHECK( cudaSetDevice( m_deviceIndex ) );
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    OTK_ERROR_CHECK( cuCtxGetCurrent( &m_cudaContext ) );

    m_loader = createDemandLoader( options );
}

void DeferredImageLoadingTest::createContext()
{
    OTK_ERROR_CHECK( optixInit() );
    OptixDeviceContextOptions options{};
    options.logCallbackFunction = contextLog;
    options.logCallbackLevel    = 4;
#ifndef NDEBUG
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OTK_ERROR_CHECK( optixDeviceContextCreate( m_cudaContext, &options, &m_context ) );

    OTK_ERROR_CHECK( cudaStreamCreate( &m_stream ) );

    OTK_ERROR_CHECK( cuCtxSetCurrent( m_cudaContext ) );
}

void DeferredImageLoadingTest::initPipelineOpts()
{
    m_pipelineOpts.usesMotionBlur         = 0;
    m_pipelineOpts.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    m_pipelineOpts.numPayloadValues       = NUM_PAYLOAD_VALUES;
    m_pipelineOpts.numAttributeValues     = NUM_ATTRIBUTE_VALUES;
    m_pipelineOpts.exceptionFlags         = OPTIX_EXCEPTION_FLAG_NONE;
    m_pipelineOpts.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    m_pipelineOpts.pipelineLaunchParamsVariableName = "params";
}

void DeferredImageLoadingTest::createModules()
{
    OptixModuleCompileOptions compileOptions{};
    compileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    otk::configModuleCompileOptions( compileOptions );
    OTK_ERROR_CHECK_LOG( optixModuleCreate( m_context, &compileOptions, &m_pipelineOpts, DeferredImageLoadingKernelsCudaText(),
                                             DeferredImageLoadingKernelsCudaSize, LOG, &LOG_SIZE, &m_module ) );
}

void DeferredImageLoadingTest::createProgramGroups()
{
    OptixProgramGroupOptions options{};
    OptixProgramGroupDesc    descs[NUM_GROUPS]{};
    descs[GROUP_RAYGEN].kind                                       = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    descs[GROUP_RAYGEN].raygen.module                              = m_module;
    descs[GROUP_RAYGEN].raygen.entryFunctionName                   = "__raygen__sampleTexture";
    descs[GROUP_MISS].kind                                         = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descs[GROUP_MISS].miss.module                                  = nullptr;
    descs[GROUP_MISS].miss.entryFunctionName                       = nullptr;
    OTK_ERROR_CHECK_LOG( optixProgramGroupCreate( m_context, descs, NUM_GROUPS, &options,
                                               LOG, &LOG_SIZE, m_groups ) );
}

void DeferredImageLoadingTest::createPipeline()
{
    const uint_t             maxTraceDepth = 1;
    OptixPipelineLinkOptions options;
    options.maxTraceDepth = maxTraceDepth;
    OTK_ERROR_CHECK_LOG( optixPipelineCreate( m_context, &m_pipelineOpts, &options, m_groups, NUM_GROUPS, LOG, &LOG_SIZE, &m_pipeline ) );

    OptixStackSizes stackSizes{};
    for( OptixProgramGroup group : m_groups )
    {
#if OPTIX_VERSION < 70700
        OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( group, &stackSizes ) );
#else
        OTK_ERROR_CHECK( optixUtilAccumulateStackSizes( group, &stackSizes, m_pipeline ) );
#endif
    }
    uint_t directCallableTraversalStackSize{};
    uint_t directCallableStateStackSize{};
    uint_t continuationStackSize{};
    OTK_ERROR_CHECK( optixUtilComputeStackSizes(
        &stackSizes, maxTraceDepth, 0, 0, &directCallableTraversalStackSize,
        &directCallableStateStackSize, &continuationStackSize ) );
    const uint_t maxTraversableDepth = 3;
    OTK_ERROR_CHECK( optixPipelineSetStackSize( m_pipeline, directCallableTraversalStackSize,
                                            directCallableStateStackSize, continuationStackSize,
                                            maxTraversableDepth ) );
}

void DeferredImageLoadingTest::buildShaderBindingTable()
{
    void*        devRayGenRecord;
    const size_t rayGenRecordSize = sizeof( RayGenSbtRecord );
    OTK_ERROR_CHECK( cudaMalloc( &devRayGenRecord, rayGenRecordSize ) );
    RayGenSbtRecord rayGenSBT;
    OTK_ERROR_CHECK( optixSbtRecordPackHeader( m_groups[GROUP_RAYGEN], &rayGenSBT ) );
    rayGenSBT.data.m_nonResidentColor = { 0.462f, 0.725f, 0.f, 1.0f };
    OTK_ERROR_CHECK( cudaMemcpy( devRayGenRecord, &rayGenSBT, rayGenRecordSize, cudaMemcpyHostToDevice ) );

    void*  devMissRecord;
    size_t missRecordSize = sizeof( MissSbtRecord );
    OTK_ERROR_CHECK( cudaMalloc( &devMissRecord, missRecordSize ) );
    MissSbtRecord missSBT;
    OTK_ERROR_CHECK( optixSbtRecordPackHeader( m_groups[GROUP_MISS], &missSBT ) );
    OTK_ERROR_CHECK( cudaMemcpy( devMissRecord, &missSBT, missRecordSize, cudaMemcpyHostToDevice ) );

    m_sbt.raygenRecord            = reinterpret_cast<CUdeviceptr>( devRayGenRecord );
    m_sbt.missRecordBase          = reinterpret_cast<CUdeviceptr>( devMissRecord );
    m_sbt.missRecordStrideInBytes = sizeof( MissSbtRecord );
    m_sbt.missRecordCount         = 1;
}

void DeferredImageLoadingTest::allocateParams()
{
    OTK_ERROR_CHECK( cudaMalloc( &m_devParams, sizeof( Params ) ) );
    m_params.m_width  = OUTPUT_WIDTH;
    m_params.m_height = OUTPUT_HEIGHT;
}

void DeferredImageLoadingTest::allocateOutput()
{
    const size_t outputSize = static_cast<size_t>( OUTPUT_WIDTH * OUTPUT_HEIGHT ) * sizeof( float4 );
    OTK_ERROR_CHECK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &m_devOutput ), outputSize ) );
    OTK_ERROR_CHECK( cuMemsetD8( reinterpret_cast<CUdeviceptr>( m_devOutput ), 0U, outputSize ) );
}

void DeferredImageLoadingTest::freeOutput()
{
    OTK_ERROR_CHECK( cuMemFree( reinterpret_cast<CUdeviceptr>( m_devOutput ) ) );
}

void DeferredImageLoadingTest::freeParams()
{
    OTK_ERROR_CHECK( cudaFree( m_devParams ) );
}

void DeferredImageLoadingTest::freeShaderBindingTable()
{
    OTK_ERROR_CHECK( cudaFree( reinterpret_cast<void*>( m_sbt.raygenRecord ) ) );
    OTK_ERROR_CHECK( cudaFree( reinterpret_cast<void*>( m_sbt.missRecordBase ) ) );
}

void DeferredImageLoadingTest::freePipeline()
{
    OTK_ERROR_CHECK( optixPipelineDestroy( m_pipeline ) );
}

void DeferredImageLoadingTest::freeProgramGroups()
{
    for( OptixProgramGroup group : m_groups )
        OTK_ERROR_CHECK( optixProgramGroupDestroy( group ) );
}

void DeferredImageLoadingTest::freeModules()
{
    OTK_ERROR_CHECK( optixModuleDestroy( m_module ) );
}

void DeferredImageLoadingTest::freeContext()
{
    OTK_ERROR_CHECK( optixDeviceContextDestroy( m_context ) );
    OTK_ERROR_CHECK( cudaStreamDestroy( m_stream ) );
}

void DeferredImageLoadingTest::freeDemandLoading()
{
    destroyDemandLoader( m_loader );
}

void DeferredImageLoadingTest::launchAndWaitForRequests()
{
    m_loader->launchPrepare( m_stream, m_params.m_context );
    OTK_ERROR_CHECK( cudaMemcpy( m_devParams, &m_params, sizeof( m_params ), cudaMemcpyHostToDevice ) );
    OTK_ERROR_CHECK( optixLaunch( m_pipeline, m_stream, reinterpret_cast<CUdeviceptr>( m_devParams ), sizeof( Params ), &m_sbt,
                              OUTPUT_WIDTH, OUTPUT_HEIGHT, /*depth=*/1 ) );
    m_loader->processRequests( m_stream, m_params.m_context ).wait();
}

}  // namespace

static demandLoading::TextureDescriptor pointSampledTexture()
{
    demandLoading::TextureDescriptor desc{};
    desc.addressMode[0]   = CU_TR_ADDRESS_MODE_CLAMP;
    desc.addressMode[1]   = CU_TR_ADDRESS_MODE_CLAMP;
    desc.filterMode       = CU_TR_FILTER_MODE_POINT;
    desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    desc.maxAnisotropy    = 1;
    return desc;
}

static imageSource::TextureInfo stockTiledImage()
{
    imageSource::TextureInfo desc;
    desc.width        = 1024;
    desc.height       = 1024;
    desc.format       = CU_AD_FORMAT_UNSIGNED_INT8;
    desc.numChannels  = 4;
    desc.numMipLevels = 1;
    desc.isValid      = true;
    desc.isTiled      = true;
    return desc;
}

static imageSource::TextureInfo stockNonTiledImage()
{
    imageSource::TextureInfo desc;
    desc.width        = 256;
    desc.height       = 256;
    desc.format       = CU_AD_FORMAT_UNSIGNED_INT8;
    desc.numChannels  = 4;
    desc.numMipLevels = 1;
    desc.isValid      = true;
    desc.isTiled      = false;
    return desc;
}

static imageSource::TextureInfo stockNonTiledMipMappedImage()
{
    imageSource::TextureInfo desc;
    desc.width        = 256;
    desc.height       = 256;
    desc.format       = CU_AD_FORMAT_UNSIGNED_INT8;
    desc.numChannels  = 4;
    desc.numMipLevels = 9;
    desc.isValid      = true;
    desc.isTiled      = false;
    return desc;
}

TEST_F( DeferredImageLoadingTest, deferredTileIsLoadedAgain )
{
    // Ignore tests for devices that do not support texture footprints
    if( !demandLoading::deviceSupportsSparseTextures( m_deviceIndex ) )
        return;

    using namespace testing;
    auto                                image{ std::make_shared<StrictMock<MockImageSource>>() };
    const demandLoading::DemandTexture& texture = m_loader->createTexture( image, pointSampledTexture() );
    EXPECT_CALL( *image, open( _ ) ).WillOnce( SetArgPointee<0>( stockTiledImage() ) );
    EXPECT_CALL( *image, getFillType() ).WillRepeatedly( Return( CU_MEMORYTYPE_HOST ) );
    EXPECT_CALL( *image, readTile( _, _, _, _ ) ).WillOnce( Return( false ) ).WillOnce( Return( true ) );
    EXPECT_CALL( *image, hasCascade() ).WillOnce( Return( false ) );
    m_params.m_output    = m_devOutput;
    m_params.m_textureId = texture.getId();

    launchAndWaitForRequests();  // Launch and get the sampler loaded.
    launchAndWaitForRequests();  // Launch and defer the tile load.
    launchAndWaitForRequests();  // Launch and satisfy the tile load.
    launchAndWaitForRequests();  // Launch and use existing tile.
}

TEST_F( DeferredImageLoadingTest, deferredMipLevelIsLoadedAgain )
{
    using namespace testing;
    auto                                image{ std::make_shared<StrictMock<MockImageSource>>() };
    const demandLoading::DemandTexture& texture = m_loader->createTexture( image, pointSampledTexture() );
    EXPECT_CALL( *image, open( _ ) ).WillOnce( SetArgPointee<0>( stockNonTiledImage() ) );
    EXPECT_CALL( *image, getFillType() ).WillRepeatedly( Return( CU_MEMORYTYPE_HOST ) );
    EXPECT_CALL( *image, readMipLevel( _, _, _, _, _ ) ).WillOnce( Return( false ) ).WillOnce( Return( true ) );
    EXPECT_CALL( *image, hasCascade() ).WillOnce( Return( false ) );
    m_params.m_output    = m_devOutput;
    m_params.m_textureId = texture.getId();

    launchAndWaitForRequests();  // Launch and get the sampler loaded.
    launchAndWaitForRequests();  // Launch and defer the mip level load.
    launchAndWaitForRequests();  // Launch and satisfy the mip level load.
    launchAndWaitForRequests();  // Launch and use existing mip level.
}

TEST_F( DeferredImageLoadingTest, deferredMipTailIsLoadedAgain )
{
    using namespace testing;
    auto                                image{ std::make_shared<StrictMock<MockImageSource>>() };
    const demandLoading::DemandTexture& texture = m_loader->createTexture( image, pointSampledTexture() );
    EXPECT_CALL( *image, open( _ ) ).WillOnce( SetArgPointee<0>( stockNonTiledMipMappedImage() ) );
    EXPECT_CALL( *image, getFillType() ).WillRepeatedly( Return( CU_MEMORYTYPE_HOST ) );
    EXPECT_CALL( *image, readMipTail( _, _, _, _, _ ) ).WillOnce( Return( false ) ).WillOnce( Return( true ) );
    EXPECT_CALL( *image, hasCascade() ).WillOnce( Return( false ) );
    m_params.m_output    = m_devOutput;
    m_params.m_textureId = texture.getId();

    launchAndWaitForRequests();  // Launch and get the sampler loaded.
    launchAndWaitForRequests();  // Launch and defer the mip level load.
    launchAndWaitForRequests();  // Launch and satisfy the mip level load.
    launchAndWaitForRequests();  // Launch and use existing mip level.
}
