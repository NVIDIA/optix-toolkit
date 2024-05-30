#include <IdRangePrinter.h>

#include <gtest/gtest.h>

#include <numeric>
#include <sstream>
#include <vector>

using namespace demandPbrtScene;

class TestIdRangePrinter : public testing::Test
{
  protected:
    std::ostringstream        str;
    std::vector<unsigned int> ids;
};

TEST_F( TestIdRangePrinter, singleId )
{
    ids.assign( { 1 } );

    str << IdRange( ids );

    EXPECT_EQ( "1", str.str() );
}

TEST_F( TestIdRangePrinter, nonConsecutiveIds )
{
    ids.assign( { 1, 3 } );

    str << IdRange( ids );

    EXPECT_EQ( "1, 3", str.str() );
}

TEST_F( TestIdRangePrinter, idRange )
{
    ids.assign( { 1, 2, 3 } );

    str << IdRange( ids );

    EXPECT_EQ( "1-3", str.str() );
}

TEST_F( TestIdRangePrinter, trailingRange )
{
    ids.assign( { 1, 4, 5, 6 } );

    str << IdRange( ids );

    EXPECT_EQ( "1, 4-6", str.str() );
}

TEST_F( TestIdRangePrinter, leadingRange )
{
    ids.assign( { 1, 2, 3, 6 } );

    str << IdRange( ids );

    EXPECT_EQ( "1-3, 6", str.str() );
}

TEST_F( TestIdRangePrinter, idRangeWithOtherIds )
{
    ids.assign( { 1, 3, 4, 6 } );

    str << IdRange( ids );

    EXPECT_EQ( "1, 3-4, 6", str.str() );
}

TEST_F( TestIdRangePrinter, largeIdRangeIncludesCount )
{
    constexpr size_t NUM_VALUES{ 50 };
    ids.resize( NUM_VALUES );
    std::iota( ids.begin(), ids.end(), 10 );

    str << IdRange( ids );

    EXPECT_EQ( "10-" + std::to_string( 10 + NUM_VALUES - 1 ) + " (" + std::to_string( NUM_VALUES ) + ')', str.str() );
}

TEST_F( TestIdRangePrinter, startsWithLargeIdRange )
{
    constexpr size_t NUM_VALUES{ 50 };
    ids.resize( NUM_VALUES );
    std::iota( ids.begin(), ids.end(), 10 );
    ids.push_back( 6633 );
    ids.push_back( 7744 );

    str << IdRange( ids );

    EXPECT_EQ( "10-" + std::to_string( 10 + NUM_VALUES - 1 ) + " (" + std::to_string( NUM_VALUES ) + "), 6633, 7744", str.str() );
}

TEST_F( TestIdRangePrinter, hasLargeIdRange )
{
    constexpr size_t NUM_VALUES{ 50 };
    ids.push_back( 6633 );
    ids.push_back( 7744 );
    ids.resize( NUM_VALUES + 2 );
    std::iota( ids.begin() + 2, ids.end(), 10 );
    ids.push_back( 8811 );
    ids.push_back( 9922 );

    str << IdRange( ids );

    EXPECT_EQ( "6633, 7744, 10-" + std::to_string( 10 + NUM_VALUES - 1 ) + " (" + std::to_string( NUM_VALUES )
                   + "), 8811, 9922",
               str.str() );
}

TEST_F( TestIdRangePrinter, endsWithLargeIdRange )
{
    constexpr size_t NUM_VALUES{ 50 };
    ids.resize( NUM_VALUES );
    ids[0] = 6633;
    ids[1] = 7744;
    std::iota( ids.begin() + 2, ids.end(), 10 );

    str << IdRange( ids );

    EXPECT_EQ( "6633, 7744, 10-" + std::to_string( 10 + NUM_VALUES - 1 - 2 ) +  //
                   " (" + std::to_string( NUM_VALUES - 2 ) + ')',
               str.str() );
}
