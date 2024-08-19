// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <MaterialBatch.h>

#include "ParamsPrinters.h"

#include <OptiXToolkit/DemandGeometry/Mocks/Matchers.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;
using namespace otk::testing;
using namespace demandPbrtScene;

namespace {

class TestMaterialBatch : public Test

{
  protected:
    void SetUp() override;
    void TearDown() override;

    CUstream         m_stream{};
    MaterialBatchPtr m_batch{ createMaterialBatch() };
};

void TestMaterialBatch::SetUp()
{
    Test::SetUp();
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    OTK_ERROR_CHECK( cuStreamCreate( &m_stream, CU_STREAM_DEFAULT ) );
}

void TestMaterialBatch::TearDown()
{
    OTK_ERROR_CHECK( cuStreamDestroy( m_stream ) );
    Test::TearDown();
}

}  // namespace

TEST_F( TestMaterialBatch, addPrimitiveMaterialRange )
{
    const uint_t materialId{ 111 };

    const uint_t startIndex = m_batch->addPrimitiveMaterialRange( 400, materialId );

    EXPECT_EQ( 0, startIndex );
}

TEST_F( TestMaterialBatch, addMultiplePrimitiveMaterialRange )
{
    const uint_t materialId{ 111 };
    const uint_t materialId2{ 222 };

    const uint_t startIndex = m_batch->addPrimitiveMaterialRange( 400, materialId );
    const uint_t nextIndex  = m_batch->addPrimitiveMaterialRange( 200, materialId2 );

    EXPECT_EQ( 0, startIndex );
    EXPECT_EQ( 1, nextIndex );
}

ListenerPredicate<const PrimitiveMaterialRange*> hasPrimitiveMaterialRange( uint_t index, uint_t primitiveIndexBegin, uint_t materialId )
{
    return [=]( MatchResultListener* listener, const PrimitiveMaterialRange* range ) {
        return hasEqualValues( listener, "num instances", PrimitiveMaterialRange{ primitiveIndexBegin, materialId }, range[index] );
    };
}

MATCHER_P2( hasDevicePrimitiveMaterialRanges, n, predicate, "" )
{
    if( arg == nullptr )
    {
        *result_listener << "PrimitiveMaterialRange array is nullptr";
        return false;
    }

    std::vector<PrimitiveMaterialRange> actualRanges;
    actualRanges.resize( n );
    OTK_ERROR_CHECK( cudaMemcpy( actualRanges.data(), arg, sizeof( PrimitiveMaterialRange ) * n, cudaMemcpyDeviceToHost ) );
    return predicate( result_listener, actualRanges.data() );
}

TEST_F( TestMaterialBatch, setPrimitiveMaterialRangesInLaunchParams )
{
    const uint_t materialId{ 111 };
    const uint_t materialId2{ 222 };
    const uint_t startIndex = m_batch->addPrimitiveMaterialRange( 400, materialId );
    const uint_t nextIndex  = m_batch->addPrimitiveMaterialRange( 200, materialId2 );
    Params       params{};

    m_batch->setLaunchParams( m_stream, params );

    EXPECT_EQ( 2U, params.numPrimitiveMaterials );
    EXPECT_NE( nullptr, params.primitiveMaterials );
    EXPECT_THAT( params.primitiveMaterials,
                 hasDevicePrimitiveMaterialRanges( 2U, hasAll( hasPrimitiveMaterialRange( startIndex, 400, materialId ),
                                                               hasPrimitiveMaterialRange( nextIndex, 200, materialId2 ) ) ) );
}

ListenerPredicate<const MaterialIndex*> hasMaterialIndex( uint_t index, uint_t numGroups, uint_t materialBegin )
{
    return [=]( MatchResultListener* listener, const MaterialIndex* indices ) {
        return hasEqualValues( listener, "num instances", MaterialIndex{ numGroups, materialBegin }, indices[index] );
    };
}

MATCHER_P2( hasDeviceMaterialIndices, n, predicate, "" )
{
    if( arg == nullptr )
    {
        *result_listener << "MaterialIndices array is nullptr";
        return false;
    }

    std::vector<MaterialIndex> actualIndices;
    actualIndices.resize( n );
    OTK_ERROR_CHECK( cudaMemcpy( actualIndices.data(), arg, sizeof( MaterialIndex ) * n, cudaMemcpyDeviceToHost ) );
    return predicate( result_listener, actualIndices.data() );
}

TEST_F( TestMaterialBatch, addMaterialIndex )
{
    const uint_t numGroups{ 3U };
    const uint_t materialBegin{ 15U };
    Params       params{};

    m_batch->addMaterialIndex( numGroups, materialBegin );

    m_batch->setLaunchParams( m_stream, params );
    EXPECT_EQ( 1U, params.numMaterialIndices );
    EXPECT_NE( nullptr, params.materialIndices );
    EXPECT_THAT( params.materialIndices, hasDeviceMaterialIndices( 1U, hasMaterialIndex( 0, numGroups, materialBegin ) ) );
}
