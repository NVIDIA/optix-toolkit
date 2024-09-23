// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <DemandPbrtScene/MaterialBatch.h>

#include "ParamsPrinters.h"

#include <DemandPbrtScene/SceneSyncState.h>

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
    SceneSyncState   m_sync;
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

    const uint_t startIndex = m_batch->addPrimitiveMaterialRange( 400, materialId, m_sync );

    EXPECT_EQ( 0, startIndex );
    ASSERT_FALSE( m_sync.primitiveMaterials.empty() );
    EXPECT_EQ( 400, m_sync.primitiveMaterials[0].primitiveEnd );
    EXPECT_EQ( materialId, m_sync.primitiveMaterials[0].materialId );
}

TEST_F( TestMaterialBatch, addMultiplePrimitiveMaterialRange )
{
    const uint_t materialId{ 111 };
    const uint_t materialId2{ 222 };

    const uint_t startIndex = m_batch->addPrimitiveMaterialRange( 400, materialId, m_sync );
    const uint_t nextIndex  = m_batch->addPrimitiveMaterialRange( 200, materialId2, m_sync );

    EXPECT_EQ( 0, startIndex );
    EXPECT_EQ( 1, nextIndex );
    ASSERT_FALSE( m_sync.primitiveMaterials.empty() );
    EXPECT_EQ( 2U, m_sync.primitiveMaterials.size() );
    EXPECT_EQ( ( PrimitiveMaterialRange{ 400, materialId } ), m_sync.primitiveMaterials[0] );
    EXPECT_EQ( ( PrimitiveMaterialRange{ 200, materialId2 } ), m_sync.primitiveMaterials[1] );
}

TEST_F( TestMaterialBatch, addMaterialIndex )
{
    const uint_t numGroups{ 3U };
    const uint_t materialBegin{ 15U };

    m_batch->addMaterialIndex( numGroups, materialBegin, m_sync );

    ASSERT_FALSE( m_sync.materialIndices.empty() );
    EXPECT_EQ( 1U, m_sync.materialIndices.size() );
    EXPECT_EQ( ( MaterialIndex{ numGroups, materialBegin } ), m_sync.materialIndices[0] );
}
