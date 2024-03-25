//
//  Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>

#include <OptiXToolkit/DemandGeometry/Mocks/MockDemandLoader.h>

#include <gmock/gmock.h>

using namespace testing;
using namespace otk::testing;

using uint_t            = unsigned int;
using MaterialLoaderPtr = std::shared_ptr<demandMaterial::MaterialLoader>;

namespace {
class TestMaterialLoader : public Test
{
  protected:
    void SetUp() override;

    MockDemandLoader  m_loader;
    MaterialLoaderPtr m_materials;
    uint_t            firstId{ 1010 };
    Expectation       expectFirstId;
};

void TestMaterialLoader::SetUp()
{
    Test::SetUp();
    m_materials   = demandMaterial::createMaterialLoader( &m_loader );
    expectFirstId = EXPECT_CALL( m_loader, createResource( 1, _, _ ) ).WillOnce( Return( firstId ) );
}

}  // namespace

TEST_F( TestMaterialLoader, removedPageIdsAreRecycled )
{
    m_materials->setRecycleProxyIds( true );
    const uint_t secondId = firstId + 1;
    EXPECT_CALL( m_loader, createResource( 1, _, _ ) ).After( expectFirstId ).WillOnce( Return( secondId ) );
    const uint_t id1 = m_materials->add();
    const uint_t id2 = m_materials->add();
    EXPECT_CALL( m_loader, unloadResource( id1 ) );
    m_materials->remove( id1 );

    const uint_t id3 = m_materials->add();

    EXPECT_EQ( firstId, id1 );
    EXPECT_EQ( secondId, id2 );
    EXPECT_EQ( id1, id3 );
    EXPECT_NE( id1, id2 );
}

TEST_F( TestMaterialLoader, alwaysNewPageIdWhenNotRecycling )
{
    m_materials->setRecycleProxyIds( false ); // the default
    const uint_t secondId = firstId + 1;
    Expectation expectSecondId = EXPECT_CALL( m_loader, createResource( 1, _, _ ) ).After( expectFirstId ).WillOnce( Return( secondId ) );
    const uint_t thirdId = secondId + 1;
    EXPECT_CALL( m_loader, createResource( 1, _, _ ) ).After( expectSecondId ).WillOnce( Return( thirdId ) );
    EXPECT_CALL( m_loader, unloadResource( _ ) ).Times( 0 );
    const uint_t id1 = m_materials->add();
    const uint_t id2 = m_materials->add();
    m_materials->remove( id1 );

    const uint_t id3 = m_materials->add();

    EXPECT_EQ( firstId, id1 );
    EXPECT_EQ( secondId, id2 );
    EXPECT_EQ( thirdId, id3 );
    EXPECT_NE( id1, id2 );
    EXPECT_NE( id1, id3 );
}

TEST_F( TestMaterialLoader, removingTwiceThrowsException )
{
    m_materials->setRecycleProxyIds( false ); // the default
    const uint_t id1 = m_materials->add();
    EXPECT_CALL( m_loader, unloadResource( _ ) ).Times( 0 );
    m_materials->remove( id1 );

    EXPECT_THROW( m_materials->remove( id1 ), std::runtime_error );
}
