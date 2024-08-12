#include <OptiXToolkit/Memory/SyncVector.h>

#include <gtest/gtest.h>


TEST( TestSyncVector, reserveIncreasesCapacity )
{
    otk::SyncVector<int> v;
    ASSERT_EQ( 0U, v.capacity() );

    v.reserve( 10 );

    EXPECT_TRUE( v.empty() );
    EXPECT_LE( 10, v.capacity() );
}
