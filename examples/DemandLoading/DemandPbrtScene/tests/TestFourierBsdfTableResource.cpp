// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <DemandPbrtScene/FourierBsdfTableResource.h>

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <cuda.h>

#include <gtest/gtest.h>

#include <vector>

using namespace demandPbrtScene;

namespace {

FourierBsdfTable arbitraryFourierTable()
{
    FourierBsdfTable table{};
    table.flags                 = 1;
    table.nMu                   = 2;
    table.nCoefficients         = 3;
    table.maxOrder              = 4;
    table.nChannels             = 3;
    table.nBases                = 1;
    table.eta                   = 1.5f;
    table.trailingByteCount     = 8U;
    table.mu                    = { -1.0f, 1.0f };
    table.cdf                   = { 0.0f, 1.0f };
    table.coefficientOffsets    = { 0, 3, 6, 9 };
    table.coefficientCounts     = { 3, 3, 3, 3 };
    table.zeroOrderCoefficients = { 0.1f, 0.2f, 0.3f };
    table.coefficients          = { 0.4f, 0.5f, 0.6f };
    return table;
}

FourierBsdfTable ceramicOrderFourierTable()
{
    FourierBsdfTable table{};
    table.flags                 = 1;
    table.nMu                   = 2;
    table.maxOrder              = 1599;
    table.nChannels             = 3;
    table.nBases                = 1;
    table.eta                   = 1.0f;
    table.mu                    = { -1.0f, 1.0f };
    table.cdf                   = { 0.0f, 1.0f, 0.0f, 1.0f };
    table.coefficientOffsets    = { 0, 4797, 9594, 14391 };
    table.coefficientCounts     = { 1599, 1599, 1599, 1599 };
    table.zeroOrderCoefficients = { 1.0f, 1.0f, 1.0f, 1.0f };
    table.nCoefficients         = 19188;
    table.coefficients.assign( static_cast<std::size_t>( table.nCoefficients ), 0.0f );
    return table;
}

class TestFourierBsdfTableResourceUpload : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        OTK_ERROR_CHECK( cuInit( 0 ) );
#if CUDA_VERSION >= 13000
        CUctxCreateParams params{};
        OTK_ERROR_CHECK( cuCtxCreate( &m_context, &params, 0, 0 ) );
#else
        OTK_ERROR_CHECK( cuCtxCreate( &m_context, 0, 0 ) );
#endif
    }

    void TearDown() override { OTK_ERROR_CHECK( cuCtxDestroy( m_context ) ); }

    CUcontext m_context{};
};

template <typename T>
std::vector<T> downloadVector( CUdeviceptr data, std::size_t size )
{
    std::vector<T> result( size );
    OTK_ERROR_CHECK( cuMemcpyDtoH( result.data(), data, result.size() * sizeof( T ) ) );
    return result;
}

}  // namespace

TEST( TestFourierBsdfTableResource, descriptorCopiesTableMetadataAndDevicePointers )
{
    const FourierBsdfTable table{ arbitraryFourierTable() };

    const FourierBsdfTableDeviceData data{ makeFourierBsdfTableDeviceData(
        table, static_cast<CUdeviceptr>( 0x1000U ), static_cast<CUdeviceptr>( 0x2000U ), static_cast<CUdeviceptr>( 0x3000U ),
        static_cast<CUdeviceptr>( 0x4000U ), static_cast<CUdeviceptr>( 0x5000U ), static_cast<CUdeviceptr>( 0x6000U ) ) };

    EXPECT_EQ( table.flags, data.flags );
    EXPECT_EQ( table.nMu, data.nMu );
    EXPECT_EQ( table.nCoefficients, data.nCoefficients );
    EXPECT_EQ( table.maxOrder, data.maxOrder );
    EXPECT_EQ( table.nChannels, data.nChannels );
    EXPECT_EQ( table.nBases, data.nBases );
    EXPECT_FLOAT_EQ( table.eta, data.eta );
    EXPECT_EQ( 8U, data.trailingByteCount );
    EXPECT_EQ( 4U, data.gridSize );
    EXPECT_EQ( static_cast<CUdeviceptr>( 0x1000U ), data.mu );
    EXPECT_EQ( static_cast<CUdeviceptr>( 0x2000U ), data.cdf );
    EXPECT_EQ( static_cast<CUdeviceptr>( 0x3000U ), data.coefficientOffsets );
    EXPECT_EQ( static_cast<CUdeviceptr>( 0x4000U ), data.coefficientCounts );
    EXPECT_EQ( static_cast<CUdeviceptr>( 0x5000U ), data.zeroOrderCoefficients );
    EXPECT_EQ( static_cast<CUdeviceptr>( 0x6000U ), data.coefficients );
}

TEST( TestFourierBsdfTableResource, descriptorCopiesCeramicOrderMetadata )
{
    const FourierBsdfTable table{ ceramicOrderFourierTable() };

    const FourierBsdfTableDeviceData data{ makeFourierBsdfTableDeviceData(
        table, static_cast<CUdeviceptr>( 0x1000U ), static_cast<CUdeviceptr>( 0x2000U ), static_cast<CUdeviceptr>( 0x3000U ),
        static_cast<CUdeviceptr>( 0x4000U ), static_cast<CUdeviceptr>( 0x5000U ), static_cast<CUdeviceptr>( 0x6000U ) ) };

    EXPECT_EQ( 1599, data.maxOrder );
    EXPECT_EQ( 3, data.nChannels );
    EXPECT_EQ( 19188, data.nCoefficients );
    EXPECT_EQ( 4U, data.gridSize );
}

TEST( TestFourierBsdfTableResource, defaultResourceHasNoDeviceDescriptor )
{
    const FourierBsdfTableDeviceResource resource;

    EXPECT_EQ( CUdeviceptr{}, resource.deviceData() );
}

TEST_F( TestFourierBsdfTableResourceUpload, uploadCopiesDescriptorAndTableStorage )
{
    const FourierBsdfTable         table{ arbitraryFourierTable() };
    FourierBsdfTableDeviceResource resource;

    resource.upload( table );

    ASSERT_NE( CUdeviceptr{}, resource.deviceData() );
    FourierBsdfTableDeviceData data{};
    OTK_ERROR_CHECK( cuMemcpyDtoH( &data, resource.deviceData(), sizeof( data ) ) );

    EXPECT_EQ( table.flags, data.flags );
    EXPECT_EQ( table.nMu, data.nMu );
    EXPECT_EQ( table.nCoefficients, data.nCoefficients );
    EXPECT_EQ( table.maxOrder, data.maxOrder );
    EXPECT_EQ( table.nChannels, data.nChannels );
    EXPECT_EQ( table.nBases, data.nBases );
    EXPECT_FLOAT_EQ( table.eta, data.eta );
    EXPECT_EQ( table.trailingByteCount, data.trailingByteCount );
    EXPECT_EQ( table.coefficientOffsets.size(), data.gridSize );
    EXPECT_EQ( table.mu, downloadVector<float>( data.mu, table.mu.size() ) );
    EXPECT_EQ( table.cdf, downloadVector<float>( data.cdf, table.cdf.size() ) );
    EXPECT_EQ( table.coefficientOffsets, downloadVector<int>( data.coefficientOffsets, table.coefficientOffsets.size() ) );
    EXPECT_EQ( table.coefficientCounts, downloadVector<int>( data.coefficientCounts, table.coefficientCounts.size() ) );
    EXPECT_EQ( table.zeroOrderCoefficients,
               downloadVector<float>( data.zeroOrderCoefficients, table.zeroOrderCoefficients.size() ) );
    EXPECT_EQ( table.coefficients, downloadVector<float>( data.coefficients, table.coefficients.size() ) );
}
