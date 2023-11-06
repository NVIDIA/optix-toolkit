//
//  Copyright (c) 2023 NVIDIA Corporation.  All rights reserved.
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

#include <OptiXToolkit/DemandGeometry/Mocks/OptixCompare.h>

#include <OptiXToolkit/Memory/BitCast.h>

#include <gtest/gtest.h>

#include <sstream>

// Use EXPECT_FALSE( a == b ), etc., instead of EXPECT_NE( a, b ) to explicitly
// exercise the comparison operator== instead of operator!=.
// For the TRUE case, the google macro will do what we want.

TEST( TestCompareOptixAabb, equalToItself )
{
    const OptixAabb one{};
    OptixAabb       two{};
    two.minX = -1.0f;
    OptixAabb three{ two };
    three.minY = -1.0f;
    OptixAabb four{ three };
    four.minZ = -1.0f;
    OptixAabb five{ four };
    five.maxX = 1.0f;
    OptixAabb six{ five };
    six.maxY = 1.0f;
    OptixAabb seven{ six };
    seven.maxZ = 1.0f;

    EXPECT_EQ( one, one );
    EXPECT_EQ( two, two );
    EXPECT_EQ( three, three );
    EXPECT_EQ( four, four );
    EXPECT_EQ( five, five );
    EXPECT_EQ( six, six );
    EXPECT_EQ( seven, seven );
    EXPECT_FALSE( one != one );
    EXPECT_FALSE( two != two );
    EXPECT_FALSE( three != three );
    EXPECT_FALSE( four != four );
    EXPECT_FALSE( five != five );
    EXPECT_FALSE( six != six );
    EXPECT_FALSE( seven != seven );
}

TEST( TestCompareOptixAabb, differentMinX )
{
    const OptixAabb one{-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f};
    const OptixAabb two{+1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f};

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixAabb, differentMinY )
{
    const OptixAabb one{-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f};
    const OptixAabb two{-1.0f, +2.0f, -3.0f, 4.0f, 5.0f, 6.0f};

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixAabb, differentMinZ )
{
    const OptixAabb one{-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f};
    const OptixAabb two{-1.0f, -2.0f, +3.0f, 4.0f, 5.0f, 6.0f};

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixAabb, differentMaxX )
{
    const OptixAabb one{-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f};
    const OptixAabb two{-1.0f, -2.0f, -3.0f, 5.0f, 5.0f, 6.0f};

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixAabb, differentMaxY )
{
    const OptixAabb one{-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f};
    const OptixAabb two{-1.0f, -2.0f, -3.0f, 4.0f, 6.0f, 6.0f};

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixAabb, differentMaxZ )
{
    const OptixAabb one{-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f};
    const OptixAabb two{-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 7.0f};

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixProgramGroupSingleModule, equalToItself )
{
    const OptixProgramGroupSingleModule one{};
    const OptixModule                   fakeModule{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixProgramGroupSingleModule two{ fakeModule, nullptr };
    const OptixProgramGroupSingleModule three{ fakeModule, "__intersection__test" };

    EXPECT_EQ( one, one );
    EXPECT_EQ( two, two );
    EXPECT_EQ( three, three );
    EXPECT_FALSE( one != one );
    EXPECT_FALSE( two != two );
    EXPECT_FALSE( three != three );
}

TEST( TestCompareOptixProgramGroupSingleModule, defaultConstructedAreEqual )
{
    const OptixProgramGroupSingleModule one{};
    const OptixProgramGroupSingleModule two{};

    EXPECT_EQ( one, two );
    EXPECT_EQ( two, one );
    EXPECT_FALSE( one != two );
    EXPECT_FALSE( two != one );
}

TEST( TestCompareOptixProgramGroupSingleModule, sameModuleNullNamesAreEqual )
{
    const OptixModule                   fakeModule{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixProgramGroupSingleModule one{ fakeModule, nullptr };
    const OptixProgramGroupSingleModule two{ fakeModule, nullptr };

    EXPECT_EQ( one, two );
    EXPECT_EQ( two, one );
    EXPECT_FALSE( one != two );
    EXPECT_FALSE( two != one );
}

TEST( TestCompareOptixProgramGroupSingleModule, differentModuleNullNamesAreNotEqual )
{
    const OptixModule                   fakeModule1{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixModule                   fakeModule2{ otk::bit_cast<OptixModule>( 2222ULL ) };
    const OptixProgramGroupSingleModule one{ fakeModule1, nullptr };
    const OptixProgramGroupSingleModule two{ fakeModule2, nullptr };

    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
}

TEST( TestCompareOptixProgramGroupSingleModule, sameModuleSameNamesAreEqual )
{
    const OptixModule                   fakeModule{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const char* const                   name1 = "__intersection__test";
    const char* const                   name2 = "__intersection__test";
    const OptixProgramGroupSingleModule one{ fakeModule, name1 };
    const OptixProgramGroupSingleModule two{ fakeModule, name2 };

    EXPECT_NE( name1, name2 );
    EXPECT_STREQ( name1, name2 );
    EXPECT_EQ( one, two );
    EXPECT_EQ( two, one );
    EXPECT_FALSE( one != two );
    EXPECT_FALSE( two != one );
}

TEST( TestCompareOptixProgramGroupSingleModule, differentModuleSameNamesAreNotEqual )
{
    const OptixModule                   fakeModule1{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixModule                   fakeModule2{ otk::bit_cast<OptixModule>( 2222ULL ) };
    const char* const                   name1 = "__intersection__test";
    const char* const                   name2 = "__intersection__test";
    const OptixProgramGroupSingleModule one{ fakeModule1, name1 };
    const OptixProgramGroupSingleModule two{ fakeModule2, name2 };

    EXPECT_NE( name1, name2 );
    EXPECT_STREQ( name1, name2 );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
}

TEST( TestCompareOptixProgramGroupSingleModule, sameModuleDifferentNamesNullNotEqual )
{
    const OptixModule                   fakeModule1{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixModule                   fakeModule2{ otk::bit_cast<OptixModule>( 2222ULL ) };
    const OptixProgramGroupSingleModule one{ fakeModule1, "__intersection__test" };
    const OptixProgramGroupSingleModule two{ fakeModule2, nullptr };

    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
}

TEST( TestCompareOptixProgramGroupHitgroup, equalToItself )
{
    const OptixModule               fakeModuleCH{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixModule               fakeModuleAH{ otk::bit_cast<OptixModule>( 2222ULL ) };
    const OptixModule               fakeModuleIS{ otk::bit_cast<OptixModule>( 3333ULL ) };
    const char* const               chName{ "__closesthit__test" };
    const char* const               ahName{ "__anyhit__test" };
    const char* const               isName{ "__intersection__test" };
    const OptixProgramGroupHitgroup one{};
    OptixProgramGroupHitgroup       two{ fakeModuleCH, nullptr, OptixModule{}, nullptr, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup three{ fakeModuleCH, chName, OptixModule{}, nullptr, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup four{ fakeModuleCH, nullptr, fakeModuleAH, nullptr, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup five{ fakeModuleCH, chName, fakeModuleAH, nullptr, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup six{ fakeModuleCH, chName, fakeModuleAH, ahName, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup seven{ fakeModuleCH, chName, fakeModuleAH, ahName, fakeModuleIS, nullptr };
    const OptixProgramGroupHitgroup eight{ fakeModuleCH, chName, fakeModuleAH, ahName, fakeModuleIS, isName };

    EXPECT_EQ( one, one );
    EXPECT_EQ( two, two );
    EXPECT_EQ( three, three );
    EXPECT_EQ( four, four );
    EXPECT_EQ( five, five );
    EXPECT_EQ( six, six );
    EXPECT_EQ( seven, seven );
    EXPECT_EQ( eight, eight );
    EXPECT_FALSE( one != one );
    EXPECT_FALSE( two != two );
    EXPECT_FALSE( three != three );
    EXPECT_FALSE( four != four );
    EXPECT_FALSE( five != five );
    EXPECT_FALSE( six != six );
    EXPECT_FALSE( seven != seven );
    EXPECT_FALSE( eight != eight );
}

TEST( TestCompareOptixProgramGroupHitgroup, sameModuleDifferentNamesNullAreNotEqual )
{
    const OptixModule               fakeModuleCH{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixModule               fakeModuleAH{ otk::bit_cast<OptixModule>( 2222ULL ) };
    const OptixModule               fakeModuleIS{ otk::bit_cast<OptixModule>( 3333ULL ) };
    const char* const               chName{ "__closesthit__test" };
    const char* const               ahName{ "__anyhit__test" };
    const char* const               isName{ "__intersection__test" };
    const OptixProgramGroupHitgroup one{ fakeModuleCH, nullptr, OptixModule{}, nullptr, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup two{ fakeModuleCH, chName, OptixModule{}, nullptr, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup three{ fakeModuleCH, nullptr, fakeModuleAH, nullptr, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup four{ fakeModuleCH, chName, fakeModuleAH, nullptr, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup five{ fakeModuleCH, chName, fakeModuleAH, ahName, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup six{ fakeModuleCH, chName, fakeModuleAH, ahName, fakeModuleIS, nullptr };
    const OptixProgramGroupHitgroup seven{ fakeModuleCH, chName, fakeModuleAH, ahName, fakeModuleIS, isName };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_NE( three, four );
    EXPECT_NE( three, five );
    EXPECT_NE( four, three );
    EXPECT_NE( five, three );
    EXPECT_NE( six, seven );
    EXPECT_NE( seven, six );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
    EXPECT_FALSE( three == four );
    EXPECT_FALSE( three == five );
    EXPECT_FALSE( four == three );
    EXPECT_FALSE( five == three );
    EXPECT_FALSE( six == seven );
    EXPECT_FALSE( seven == six );
}

TEST( TestCompareOptixProgramGroupHitgroup, sameModuleSameNameDifferentNamePointersAreEqual )
{
    const OptixModule               fakeModuleCH{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixModule               fakeModuleAH{ otk::bit_cast<OptixModule>( 2222ULL ) };
    const OptixModule               fakeModuleIS{ otk::bit_cast<OptixModule>( 3333ULL ) };
    const char* const               chName1{ "__closesthit__test" };
    const char* const               chName2{ "__closesthit__test" };
    const char* const               ahName1{ "__anyhit__test" };
    const char* const               ahName2{ "__anyhit__test" };
    const char* const               isName1{ "__intersection__test" };
    const char* const               isName2{ "__intersection__test" };
    const OptixProgramGroupHitgroup one{ fakeModuleCH, chName1, OptixModule{}, nullptr, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup two{ fakeModuleCH, chName2, OptixModule{}, nullptr, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup three{ fakeModuleCH, chName1, fakeModuleAH, nullptr, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup four{ fakeModuleCH, chName2, fakeModuleAH, nullptr, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup five{ fakeModuleCH, chName1, fakeModuleAH, ahName1, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup six{ fakeModuleCH, chName2, fakeModuleAH, ahName2, OptixModule{}, nullptr };
    const OptixProgramGroupHitgroup seven{ fakeModuleCH, chName1, fakeModuleAH, ahName1, fakeModuleIS, isName1 };
    const OptixProgramGroupHitgroup eight{ fakeModuleCH, chName2, fakeModuleAH, ahName2, fakeModuleIS, isName2 };

    EXPECT_NE( chName1, chName2 );
    EXPECT_NE( ahName1, ahName2 );
    EXPECT_NE( isName1, isName2 );
    EXPECT_EQ( one, two );
    EXPECT_EQ( two, one );
    EXPECT_EQ( three, four );
    EXPECT_EQ( four, three );
    EXPECT_EQ( five, six );
    EXPECT_EQ( six, five );
    EXPECT_EQ( seven, eight );
    EXPECT_EQ( eight, seven );
    EXPECT_FALSE( one != two );
    EXPECT_FALSE( two != one );
    EXPECT_FALSE( three != four );
    EXPECT_FALSE( four != three );
    EXPECT_FALSE( five != six );
    EXPECT_FALSE( six != five );
    EXPECT_FALSE( seven != eight );
    EXPECT_FALSE( eight != seven );
}

TEST( TestCompareOptixProgramGroupCallables, equalToItself )
{
    const OptixModule                fakeModuleDC{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixModule                fakeModuleCC{ otk::bit_cast<OptixModule>( 2222ULL ) };
    const char* const                nameDC{ "__directcallable__test" };
    const char* const                nameCC{ "__continuationcallable__test" };
    const OptixProgramGroupCallables one{};
    const OptixProgramGroupCallables two{ fakeModuleDC, nullptr, OptixModule{}, nullptr };
    const OptixProgramGroupCallables three{ fakeModuleDC, nameDC, OptixModule{}, nullptr };
    const OptixProgramGroupCallables four{ fakeModuleDC, nameDC, fakeModuleCC, nullptr };
    const OptixProgramGroupCallables five{ fakeModuleDC, nameDC, fakeModuleCC, nameCC };

    EXPECT_EQ( one, one );
    EXPECT_EQ( two, two );
    EXPECT_EQ( three, three );
    EXPECT_EQ( four, four );
    EXPECT_EQ( five, five );
    EXPECT_FALSE( one != one );
    EXPECT_FALSE( two != two );
    EXPECT_FALSE( three != three );
    EXPECT_FALSE( four != four );
    EXPECT_FALSE( five != five );
}

TEST( TestCompareOptixProgramGroupCallables, sameModuleSameNameDifferentNamePointersAreEqual )
{
    const OptixModule                fakeModuleDC{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixModule                fakeModuleCC{ otk::bit_cast<OptixModule>( 2222ULL ) };
    const char* const                nameDC1{ "__directcallable__test" };
    const char* const                nameDC2{ "__directcallable__test" };
    const char* const                nameCC1{ "__continuationcallable__test" };
    const char* const                nameCC2{ "__continuationcallable__test" };
    const OptixProgramGroupCallables one{ fakeModuleDC, nameDC1, OptixModule{}, nullptr };
    const OptixProgramGroupCallables two{ fakeModuleDC, nameDC2, OptixModule{}, nullptr };
    const OptixProgramGroupCallables three{ fakeModuleDC, nameDC1, fakeModuleCC, nameCC1 };
    const OptixProgramGroupCallables four{ fakeModuleDC, nameDC2, fakeModuleCC, nameCC2 };

    EXPECT_NE( nameDC1, nameDC2 );
    EXPECT_STREQ( nameDC1, nameDC2 );
    EXPECT_NE( nameCC1, nameCC2 );
    EXPECT_STREQ( nameCC1, nameCC2 );
    EXPECT_EQ( one, two );
    EXPECT_EQ( two, one );
    EXPECT_EQ( three, four );
    EXPECT_EQ( four, three );
    EXPECT_FALSE( one != two );
    EXPECT_FALSE( two != one );
    EXPECT_FALSE( three != four );
    EXPECT_FALSE( four != three );
}

TEST( TestCompareOptixProgramGroupCallables, sameModuleDifferentNamesAreNotEqual )
{
    const OptixModule                fakeModuleDC{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixModule                fakeModuleCC{ otk::bit_cast<OptixModule>( 2222ULL ) };
    const char* const                nameDC1{ "__directcallable__test1" };
    const char* const                nameDC2{ "__directcallable__test2" };
    const char* const                nameCC1{ "__continuationcallable__test1" };
    const char* const                nameCC2{ "__continuationcallable__test2" };
    const OptixProgramGroupCallables one{ fakeModuleDC, nameDC1, OptixModule{}, nullptr };
    const OptixProgramGroupCallables two{ fakeModuleDC, nameDC2, OptixModule{}, nullptr };
    const OptixProgramGroupCallables three{ fakeModuleDC, nameDC1, fakeModuleCC, nameCC1 };
    const OptixProgramGroupCallables four{ fakeModuleDC, nameDC2, fakeModuleCC, nameCC2 };

    EXPECT_NE( std::string{ nameDC1 }, nameDC2 );
    EXPECT_NE( std::string{ nameCC1 }, nameCC2 );
    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_NE( three, four );
    EXPECT_NE( four, three );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
    EXPECT_FALSE( three == four );
    EXPECT_FALSE( four == three );
}

TEST( TestCompareOptixProgramGroupCallables, differentModuleSameNamesAreNotEqual )
{
    const OptixModule                fakeModuleDC1{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixModule                fakeModuleDC2{ otk::bit_cast<OptixModule>( 2222ULL ) };
    const OptixModule                fakeModuleCC1{ otk::bit_cast<OptixModule>( 3333ULL ) };
    const OptixModule                fakeModuleCC2{ otk::bit_cast<OptixModule>( 4444ULL ) };
    const char* const                nameDC{ "__directcallable__test1" };
    const char* const                nameCC{ "__continuationcallable__test1" };
    const OptixProgramGroupCallables one{ fakeModuleDC1, nameDC, OptixModule{}, nullptr };
    const OptixProgramGroupCallables two{ fakeModuleDC2, nameDC, OptixModule{}, nullptr };
    const OptixProgramGroupCallables three{ fakeModuleDC1, nameDC, fakeModuleCC1, nameCC };
    const OptixProgramGroupCallables four{ fakeModuleDC2, nameDC, fakeModuleCC1, nameCC };
    const OptixProgramGroupCallables five{ fakeModuleDC1, nameDC, fakeModuleCC2, nameCC };
    const OptixProgramGroupCallables six{ fakeModuleDC2, nameDC, fakeModuleCC2, nameCC };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_NE( three, four );
    EXPECT_NE( three, five );
    EXPECT_NE( three, six );
    EXPECT_NE( four, three );
    EXPECT_NE( four, five );
    EXPECT_NE( four, six );
    EXPECT_NE( five, three );
    EXPECT_NE( five, four );
    EXPECT_NE( five, six );
    EXPECT_NE( six, three );
    EXPECT_NE( six, four );
    EXPECT_NE( six, five );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
    EXPECT_FALSE( three == four );
    EXPECT_FALSE( three == five );
    EXPECT_FALSE( three == six );
    EXPECT_FALSE( four == three );
    EXPECT_FALSE( four == five );
    EXPECT_FALSE( four == six );
    EXPECT_FALSE( five == three );
    EXPECT_FALSE( five == four );
    EXPECT_FALSE( five == six );
    EXPECT_FALSE( six == three );
    EXPECT_FALSE( six == four );
    EXPECT_FALSE( six == five );
}

static OptixProgramGroupDesc rayGenDesc( OptixProgramGroupSingleModule raygen )
{
    OptixProgramGroupDesc desc{};
    desc.kind   = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen = raygen;
    return desc;
}

static OptixProgramGroupDesc missDesc( OptixProgramGroupSingleModule miss )
{
    OptixProgramGroupDesc desc{};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss = miss;
    return desc;
}

static OptixProgramGroupDesc exceptDesc( OptixProgramGroupSingleModule exception )
{
    OptixProgramGroupDesc desc{};
    desc.kind      = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    desc.exception = exception;
    return desc;
}

static OptixProgramGroupDesc callDesc( OptixProgramGroupCallables callables )
{
    OptixProgramGroupDesc desc{};
    desc.kind      = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    desc.callables = callables;
    return desc;
}

static OptixProgramGroupDesc hitDesc( OptixProgramGroupHitgroup hitgroup )
{
    OptixProgramGroupDesc desc{};
    desc.kind     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc.hitgroup = hitgroup;
    return desc;
}

TEST( TestCompareOptixProgramGroupDesc, equalToItself )
{
    const OptixModule           fakeModuleRG{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixModule           fakeModuleMS{ otk::bit_cast<OptixModule>( 2222ULL ) };
    const OptixModule           fakeModuleEX{ otk::bit_cast<OptixModule>( 3333ULL ) };
    const OptixModule           fakeModuleCL{ otk::bit_cast<OptixModule>( 4444ULL ) };
    const OptixModule           fakeModuleCH{ otk::bit_cast<OptixModule>( 5555ULL ) };
    const OptixModule           fakeModuleAH{ otk::bit_cast<OptixModule>( 6666ULL ) };
    const OptixModule           fakeModuleIS{ otk::bit_cast<OptixModule>( 7777ULL ) };
    const char* const           nameRG{ "__raygen__test" };
    const char* const           nameMS{ "__miss__test" };
    const char* const           nameEX{ "__exception__test" };
    const char* const           nameCL{ "__callable__test" };
    const char* const           nameCH{ "__closesthit__test" };
    const char* const           nameAH{ "__anyhit__test" };
    const char* const           nameIS{ "__intersection__test" };
    const OptixProgramGroupDesc one{};
    OptixProgramGroupDesc       two{};
    two.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    OptixProgramGroupDesc three{};
    three.kind  = OPTIX_PROGRAM_GROUP_KIND_MISS;
    three.flags = 10U;
    const OptixProgramGroupDesc four{ rayGenDesc( { fakeModuleRG, nameRG } ) };
    const OptixProgramGroupDesc five{ missDesc( { fakeModuleMS, nameMS } ) };
    const OptixProgramGroupDesc six{ exceptDesc( { fakeModuleEX, nameEX } ) };
    const OptixProgramGroupDesc seven{ callDesc( { fakeModuleCL, nameCL, OptixModule{}, nullptr } ) };
    const OptixProgramGroupDesc eight{ hitDesc( { fakeModuleCH, nameCH, OptixModule{}, nullptr, OptixModule{}, nullptr } ) };
    const OptixProgramGroupDesc nine{ hitDesc( { fakeModuleCH, nameCH, fakeModuleAH, nameAH, OptixModule{}, nullptr } ) };
    const OptixProgramGroupDesc ten{ hitDesc( { fakeModuleCH, nameCH, fakeModuleAH, nameAH, fakeModuleIS, nameIS } ) };

    EXPECT_EQ( one, one );
    EXPECT_EQ( two, two );
    EXPECT_EQ( three, three );
    EXPECT_EQ( four, four );
    EXPECT_EQ( five, five );
    EXPECT_EQ( six, six );
    EXPECT_EQ( seven, seven );
    EXPECT_EQ( eight, eight );
    EXPECT_EQ( nine, nine );
    EXPECT_EQ( ten, ten );
    EXPECT_FALSE( one != one );
    EXPECT_FALSE( two != two );
    EXPECT_FALSE( three != three );
    EXPECT_FALSE( four != four );
    EXPECT_FALSE( five != five );
    EXPECT_FALSE( six != six );
    EXPECT_FALSE( seven != seven );
    EXPECT_FALSE( eight != eight );
    EXPECT_FALSE( nine != nine );
    EXPECT_FALSE( ten != ten );
}

TEST( TestCompareOptixProgramGroupDesc, differentDescKindsAreNotEqual )
{
    OptixProgramGroupDesc one{};
    one.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    OptixProgramGroupDesc two{};
    two.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixProgramGroupDesc, differentFlagsAreNotEqual )
{
    OptixProgramGroupDesc one{};
    one.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    OptixProgramGroupDesc two{};
    two.kind  = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    two.flags = 10U;

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixInstance, equalToItself )
{
    const OptixInstance one{};
    OptixInstance       two{};
    two.transform[0] = 1.0f;
    OptixInstance three{ two };
    two.instanceId = 10U;
    OptixInstance four{ three };
    three.sbtOffset = 10U;
    OptixInstance five{ four };
    four.visibilityMask = 10U;
    OptixInstance six{ five };
    six.flags = 10U;
    OptixInstance seven{ six };
    seven.traversableHandle = OptixTraversableHandle{ 10U };

    EXPECT_EQ( one, one );
    EXPECT_EQ( two, two );
    EXPECT_EQ( three, three );
    EXPECT_EQ( four, four );
    EXPECT_EQ( five, five );
    EXPECT_EQ( six, six );
    EXPECT_EQ( seven, seven );
    EXPECT_FALSE( one != one );
    EXPECT_FALSE( two != two );
    EXPECT_FALSE( three != three );
    EXPECT_FALSE( four != four );
    EXPECT_FALSE( five != five );
    EXPECT_FALSE( six != six );
    EXPECT_FALSE( seven != seven );
}

TEST( TestCompareOptixInstance, differentTransformsAreNotEqual )
{
    OptixInstance instances[12]{};
    for( int i = 0; i < 12; ++i )
    {
        instances[i].transform[i] = 1.0f;
    }

    for( int i = 0; i < 12; ++i )
    {
        for( int j = i + 1; j < 12; ++j )
        {
            EXPECT_NE( instances[i], instances[j] );
            EXPECT_NE( instances[j], instances[i] );
            EXPECT_FALSE( instances[i] == instances[j] );
            EXPECT_FALSE( instances[j] == instances[i] );
        }
    }
}

TEST( TestCompareOptixInstance, differentInstanceIdsAreNotEqual )
{
    OptixInstance one{};
    one.instanceId = 10U;
    OptixInstance two{};
    two.instanceId = 11U;

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixInstance, differentSbtOffsetsAreNotEqual )
{
    OptixInstance one{};
    one.sbtOffset = 10U;
    OptixInstance two{};
    two.sbtOffset = 11U;

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixInstance, differentVisibilityMasksAreNotEqual )
{
    OptixInstance one{};
    one.visibilityMask = 10U;
    OptixInstance two{};
    two.visibilityMask = 11U;

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixInstance, differentFlagsAreNotEqual )
{
    OptixInstance one{};
    one.flags = 10U;
    OptixInstance two{};
    two.flags = 11U;

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixInstance, differentTraversablesAreNotEqual )
{
    OptixInstance one{};
    one.traversableHandle = OptixTraversableHandle{ 10U };
    OptixInstance two{};
    two.traversableHandle = OptixTraversableHandle{ 11U };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixInstance, differentPaddingAreEqual )
{
    OptixInstance one{};
    one.pad[0] = 1U;
    OptixInstance two{};
    two.pad[0] = 2U;
    OptixInstance three{};
    three.pad[0] = 1U;
    three.pad[1] = 1U;
    OptixInstance four{};
    four.pad[0] = 1U;
    four.pad[1] = 2U;

    EXPECT_EQ( one, two );
    EXPECT_EQ( one, three );
    EXPECT_EQ( one, four );
    EXPECT_EQ( two, one );
    EXPECT_EQ( two, three );
    EXPECT_EQ( two, four );
    EXPECT_EQ( three, one );
    EXPECT_EQ( three, two );
    EXPECT_EQ( three, four );
    EXPECT_EQ( four, one );
    EXPECT_EQ( four, two );
    EXPECT_EQ( four, three );
    EXPECT_FALSE( one != two );
    EXPECT_FALSE( one != three );
    EXPECT_FALSE( one != four );
    EXPECT_FALSE( two != one );
    EXPECT_FALSE( two != three );
    EXPECT_FALSE( two != four );
    EXPECT_FALSE( three != one );
    EXPECT_FALSE( three != two );
    EXPECT_FALSE( three != four );
    EXPECT_FALSE( four != one );
    EXPECT_FALSE( four != two );
    EXPECT_FALSE( four != three );
}

TEST( TestCompareOptixPipelineCompileOptions, isEqualToItself )
{
    const OptixPipelineCompileOptions one{};
    OptixPipelineCompileOptions       two{};
    two.usesMotionBlur = 1;
    OptixPipelineCompileOptions three{ two };
    three.traversableGraphFlags = 1U;
    OptixPipelineCompileOptions four{ three };
    four.numPayloadValues = 1;
    OptixPipelineCompileOptions five{ four };
    five.numAttributeValues = 1;
    OptixPipelineCompileOptions six{ five };
    six.exceptionFlags = 1U;
    OptixPipelineCompileOptions seven{ six };
    seven.pipelineLaunchParamsVariableName = "params";
    OptixPipelineCompileOptions eight{ seven };
    eight.usesPrimitiveTypeFlags = 1U;
#if OPTIX_VERSION >= 70600
    OptixPipelineCompileOptions nine{ eight };
    nine.allowOpacityMicromaps = 1;
#endif

    EXPECT_EQ( one, one );
    EXPECT_EQ( two, two );
    EXPECT_EQ( three, three );
    EXPECT_EQ( four, four );
    EXPECT_EQ( five, five );
    EXPECT_EQ( six, six );
    EXPECT_EQ( seven, seven );
    EXPECT_EQ( eight, eight );
#if OPTIX_VERSION >= 70600
    EXPECT_EQ( nine, nine );
#endif
}

TEST( TestCompareOptixPipelineCompileOptions, sameParamNameDifferentNamePointersAreEqual )
{
    OptixPipelineCompileOptions one{};
    one.pipelineLaunchParamsVariableName = "params";
    OptixPipelineCompileOptions two{};
    two.pipelineLaunchParamsVariableName = "params";

    EXPECT_NE( one.pipelineLaunchParamsVariableName, two.pipelineLaunchParamsVariableName );
    EXPECT_STREQ( one.pipelineLaunchParamsVariableName, two.pipelineLaunchParamsVariableName );
    EXPECT_EQ( one, two );
    EXPECT_EQ( two, one );
    EXPECT_FALSE( one != two );
    EXPECT_FALSE( two != one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentMotionBlurAreNotEqual )
{
    OptixPipelineCompileOptions one{};
    OptixPipelineCompileOptions two{};
    two.usesMotionBlur = 1;

    EXPECT_NE( one.usesMotionBlur, two.usesMotionBlur );
    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentTraversableFlagsAreNotEqual )
{
    OptixPipelineCompileOptions one{};
    one.traversableGraphFlags = 1U;
    OptixPipelineCompileOptions two{};
    two.traversableGraphFlags = 2U;

    EXPECT_NE( one.traversableGraphFlags, two.traversableGraphFlags );
    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentNumPayloadValuesAreNotEqual )
{
    OptixPipelineCompileOptions one{};
    one.numPayloadValues = 1;
    OptixPipelineCompileOptions two{};
    two.numPayloadValues = 2;

    EXPECT_NE( one.numPayloadValues, two.numPayloadValues );
    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentNumAttributeValuesAreNotEqual )
{
    OptixPipelineCompileOptions one{};
    one.numAttributeValues = 1;
    OptixPipelineCompileOptions two{};
    two.numAttributeValues = 2;

    EXPECT_NE( one.numAttributeValues, two.numAttributeValues );
    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentExceptionFlagsAreNotEqual )
{
    OptixPipelineCompileOptions one{};
    one.exceptionFlags = 1U;
    OptixPipelineCompileOptions two{};
    two.exceptionFlags = 2U;

    EXPECT_NE( one.exceptionFlags, two.exceptionFlags );
    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentParamNamesAreNotEqual )
{
    OptixPipelineCompileOptions one{};
    one.pipelineLaunchParamsVariableName = "param1";
    OptixPipelineCompileOptions two{};
    two.pipelineLaunchParamsVariableName = "param2";

    EXPECT_NE( one.pipelineLaunchParamsVariableName, two.pipelineLaunchParamsVariableName );
    EXPECT_STRNE( one.pipelineLaunchParamsVariableName, two.pipelineLaunchParamsVariableName );
    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentPrimitiveTypeFlagsAreNotEqual )
{
    OptixPipelineCompileOptions one{};
    one.usesPrimitiveTypeFlags = 1U;
    OptixPipelineCompileOptions two{};
    two.usesPrimitiveTypeFlags = 2U;

    EXPECT_NE( one.usesPrimitiveTypeFlags, two.usesPrimitiveTypeFlags );
    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

#if OPTIX_VERSION >= 70600
TEST( TestCompareOptixPipelineCompileOptions, differentOpacityMicromapsAreNotEqual )
{
    OptixPipelineCompileOptions one{};
    OptixPipelineCompileOptions two{};
    two.allowOpacityMicromaps = 1;

    EXPECT_NE( one.allowOpacityMicromaps, two.allowOpacityMicromaps );
    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}
#endif

TEST( TestOutputOptixAabb, defaultConstructed )
{
    std::ostringstream str;
    OptixAabb          value{};

    str << value;

    EXPECT_EQ( "{ min(0, 0, 0), max(0, 0, 0) }", str.str() );
}

TEST( TestOutputOptixAabb, hasValues )
{
    std::ostringstream str;
    OptixAabb          value{ -1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f };

    str << value;

    EXPECT_EQ( "{ min(-1, -2, -3), max(4, 5, 6) }", str.str() );
}

TEST( TestOutputOptixInstance, defaultConstructed )
{
    std::ostringstream str;
    OptixInstance      value{};

    str << value;

    EXPECT_EQ( "Instance{ [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 0, 0, 0, 0, 0 }", str.str() );
}

TEST( TestOutputOptixPipelineCompileOptions, defaultConstructed )
{
    std::ostringstream          str;
    OptixPipelineCompileOptions value{};

    str << value;

#if OPTIX_VERSION >= 70600
    EXPECT_EQ( "PipelineCompileOptions{ 0, 0 (0x00000000), 0, 0, 0 (0x00000000), nullptr, 0 (0x00000000), 0 }", str.str() );
#else
    EXPECT_EQ( "PipelineCompileOptions{ 0, 0 (0x00000000), 0, 0, 0 (0x00000000), nullptr, 0 (0x00000000) }", str.str() );
#endif
}

TEST( TestOutputOptixPipelineCompileOptions, hasValues )
{
    std::ostringstream          str;
    OptixPipelineCompileOptions value;
    value.usesMotionBlur                   = 1;
    value.traversableGraphFlags            = 0xf00dU;
    value.numPayloadValues                 = 2;
    value.numAttributeValues               = 3;
    value.exceptionFlags                   = 0xbaadU;
    value.pipelineLaunchParamsVariableName = "variable_name";
    value.usesPrimitiveTypeFlags           = 0xf337U;
#if OPTIX_VERSION >= 70600
    value.allowOpacityMicromaps = 4;
#endif

    str << value;

#if OPTIX_VERSION >= 70600
    EXPECT_EQ(
        "PipelineCompileOptions{ 1, 61453 (0x0000f00d), 2, 3, 47789 (0x0000baad), variable_name, 62263 (0x0000f337), 4 "
        "}",
        str.str() );
#else
    EXPECT_EQ(
        "PipelineCompileOptions{ 1, 61453 (0x0000f00d), 2, 3, 47789 (0x0000baad), variable_name, 62263 (0x0000f337) }", str.str() );
#endif
}

TEST( TestOutputOptixProgramGroupCallables, defaultConstructed )
{
    std::ostringstream         str;
    OptixProgramGroupCallables value{};

    str << value;

    EXPECT_EQ( "Callables{ DC{ 0000000000000000, nullptr }, CC{ 0000000000000000, nullptr } }", str.str() );
}

TEST( TestOutputOptixProgramGroupCallables, hasValues )
{
    std::ostringstream         str;
    OptixProgramGroupCallables value;
    value.moduleDC            = otk::bit_cast<OptixModule>( 0x1111deadbeefULL );
    value.entryFunctionNameDC = "__direct_callable__fn";
    value.moduleCC            = otk::bit_cast<OptixModule>( 0x2222deadbeefULL );
    value.entryFunctionNameCC = "__continuation_callable__fn";

    str << value;

    EXPECT_EQ(
        "Callables{ DC{ 00001111DEADBEEF, __direct_callable__fn }, CC{ 00002222DEADBEEF, __continuation_callable__fn } "
        "}",
        str.str() );
}
TEST( TestOutputOptixProgramGroupHitgroup, defaultConstructed )
{
    std::ostringstream        str;
    OptixProgramGroupHitgroup value{};

    str << value;

    EXPECT_EQ(
        "HitGroup{ IS{ 0000000000000000, nullptr }, AH{ 0000000000000000, nullptr }, CH{ 0000000000000000, nullptr } }",
        str.str() );
}

TEST( TestOutputOptixProgramGroupHitgroup, hasValues )
{
    std::ostringstream        str;
    OptixProgramGroupHitgroup value;
    value.moduleIS            = otk::bit_cast<OptixModule>( 0x1111deadbeefULL );
    value.entryFunctionNameIS = "__intersection__mesh";
    value.moduleAH            = otk::bit_cast<OptixModule>( 0x2222deadbeefULL );
    value.entryFunctionNameAH = "__anyhit__mesh";
    value.moduleCH            = otk::bit_cast<OptixModule>( 0x3333deadbeefULL );
    value.entryFunctionNameCH = "__closesthit__mesh";

    str << value;

    EXPECT_EQ(
        "HitGroup{ IS{ 00001111DEADBEEF, __intersection__mesh }, AH{ 00002222DEADBEEF, __anyhit__mesh }, "
        "CH{ 00003333DEADBEEF, __closesthit__mesh } }",
        str.str() );
}

TEST( TestOutputOptixProgramGroupSingleModule, defaultConstructed )
{
    std::ostringstream            str;
    OptixProgramGroupSingleModule value{};

    str << value;

    EXPECT_EQ( "SingleModule{ 0000000000000000, nullptr }", str.str() );
}

TEST( TestOutputOptixProgramGroupSingleModule, hasValues )
{
    std::ostringstream            str;
    OptixProgramGroupSingleModule value;
    value.module            = otk::bit_cast<OptixModule>( 0xdeadbeefULL );
    value.entryFunctionName = "__raygen__parallel";

    str << value;

    EXPECT_EQ( "SingleModule{ 00000000DEADBEEF, __raygen__parallel }", str.str() );
}
