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

// Use EXPECT_FALSE( a == b ), etc., instead of EXPECT_NE( a, b ) to explicitly
// exercise the comparison operator== instead of operator!=.
// For the TRUE case, the google macro will do what we want.

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
    const OptixProgramGroupSingleModule one{ fakeModule };
    const OptixProgramGroupSingleModule two{ fakeModule };

    EXPECT_EQ( one, two );
    EXPECT_EQ( two, one );
    EXPECT_FALSE( one != two );
    EXPECT_FALSE( two != one );
}

TEST( TestCompareOptixProgramGroupSingleModule, differentModuleNullNamesAreNotEqual )
{
    const OptixModule                   fakeModule1{ otk::bit_cast<OptixModule>( 1111ULL ) };
    const OptixModule                   fakeModule2{ otk::bit_cast<OptixModule>( 2222ULL ) };
    const OptixProgramGroupSingleModule one{ fakeModule1 };
    const OptixProgramGroupSingleModule two{ fakeModule2 };

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
    const OptixProgramGroupHitgroup two{ fakeModuleCH };
    const OptixProgramGroupHitgroup three{ fakeModuleCH, chName };
    const OptixProgramGroupHitgroup four{ fakeModuleCH, nullptr, fakeModuleAH };
    const OptixProgramGroupHitgroup five{ fakeModuleCH, chName, fakeModuleAH };
    const OptixProgramGroupHitgroup six{ fakeModuleCH, chName, fakeModuleAH, ahName };
    const OptixProgramGroupHitgroup seven{ fakeModuleCH, chName, fakeModuleAH, ahName, fakeModuleIS };
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
    const OptixProgramGroupHitgroup one{ fakeModuleCH };
    const OptixProgramGroupHitgroup two{ fakeModuleCH, chName };
    const OptixProgramGroupHitgroup three{ fakeModuleCH, nullptr, fakeModuleAH };
    const OptixProgramGroupHitgroup four{ fakeModuleCH, chName, fakeModuleAH };
    const OptixProgramGroupHitgroup five{ fakeModuleCH, chName, fakeModuleAH, ahName };
    const OptixProgramGroupHitgroup six{ fakeModuleCH, chName, fakeModuleAH, ahName, fakeModuleIS };
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
    const OptixProgramGroupHitgroup one{ fakeModuleCH, chName1 };
    const OptixProgramGroupHitgroup two{ fakeModuleCH, chName2 };
    const OptixProgramGroupHitgroup three{ fakeModuleCH, chName1, fakeModuleAH };
    const OptixProgramGroupHitgroup four{ fakeModuleCH, chName2, fakeModuleAH };
    const OptixProgramGroupHitgroup five{ fakeModuleCH, chName1, fakeModuleAH, ahName1 };
    const OptixProgramGroupHitgroup six{ fakeModuleCH, chName2, fakeModuleAH, ahName2 };
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
    const OptixProgramGroupCallables two{ fakeModuleDC };
    const OptixProgramGroupCallables three{ fakeModuleDC, nameDC };
    const OptixProgramGroupCallables four{ fakeModuleDC, nameDC, fakeModuleCC };
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
    const OptixProgramGroupCallables one{ fakeModuleDC, nameDC1 };
    const OptixProgramGroupCallables two{ fakeModuleDC, nameDC2 };
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
    const OptixProgramGroupCallables one{ fakeModuleDC, nameDC1 };
    const OptixProgramGroupCallables two{ fakeModuleDC, nameDC2 };
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
    const OptixProgramGroupCallables one{ fakeModuleDC1, nameDC };
    const OptixProgramGroupCallables two{ fakeModuleDC2, nameDC };
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
    const OptixProgramGroupDesc two{ OPTIX_PROGRAM_GROUP_KIND_RAYGEN };
    const OptixProgramGroupDesc three{ OPTIX_PROGRAM_GROUP_KIND_MISS, 10U };
    const OptixProgramGroupDesc four{ rayGenDesc( { fakeModuleRG, nameRG } ) };
    const OptixProgramGroupDesc five{ missDesc( { fakeModuleMS, nameMS } ) };
    const OptixProgramGroupDesc six{ exceptDesc( { fakeModuleEX, nameEX } ) };
    const OptixProgramGroupDesc seven{ callDesc( { fakeModuleCL, nameCL } ) };
    const OptixProgramGroupDesc eight{ hitDesc( { fakeModuleCH, nameCH } ) };
    const OptixProgramGroupDesc nine{ hitDesc( { fakeModuleCH, nameCH, fakeModuleAH, nameAH } ) };
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
    const OptixProgramGroupDesc one{ OPTIX_PROGRAM_GROUP_KIND_RAYGEN };
    const OptixProgramGroupDesc two{ OPTIX_PROGRAM_GROUP_KIND_MISS };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixProgramGroupDesc, differentFlagsAreNotEqual )
{
    const OptixProgramGroupDesc one{ OPTIX_PROGRAM_GROUP_KIND_RAYGEN };
    const OptixProgramGroupDesc two{ OPTIX_PROGRAM_GROUP_KIND_RAYGEN, 10U };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixInstance, equalToItself )
{
    const OptixInstance one{};
    const OptixInstance two{ { 1.0f } };
    const OptixInstance three{ { 1.0f }, 10U };
    const OptixInstance four{ { 1.0f }, 10U, 10U };
    const OptixInstance five{ { 1.0f }, 10U, 10U, 10U };
    const OptixInstance six{ { 1.0f }, 10U, 10U, 10U, 10U };
    const OptixInstance seven{ { 1.0f }, 10U, 10U, 10U, 10U, OptixTraversableHandle{ 10U } };

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
    const OptixInstance one{ {}, 10U };
    const OptixInstance two{ {}, 11U };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixInstance, differentSbtOffsetsAreNotEqual )
{
    const OptixInstance one{ {}, 0U, 10U };
    const OptixInstance two{ {}, 0U, 11U };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixInstance, differentVisibilityMasksAreNotEqual )
{
    const OptixInstance one{ {}, 0U, 0U, 10U };
    const OptixInstance two{ {}, 0U, 0U, 11U };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixInstance, differentFlagsAreNotEqual )
{
    const OptixInstance one{ {}, 0U, 0U, 0U, 10U };
    const OptixInstance two{ {}, 0U, 0U, 0U, 11U };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixInstance, differentTraversablesAreNotEqual )
{
    const OptixInstance one{ {}, 0U, 0U, 0U, 0U, OptixTraversableHandle{ 10U } };
    const OptixInstance two{ {}, 0U, 0U, 0U, 0U, OptixTraversableHandle{ 11U } };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixInstance, differentPaddingAreEqual )
{
    const OptixInstance one{ {}, 0U, 0U, 0U, 0U, OptixTraversableHandle{ 10U }, 1U };
    const OptixInstance two{ {}, 0U, 0U, 0U, 0U, OptixTraversableHandle{ 10U }, 2U };
    const OptixInstance three{ {}, 0U, 0U, 0U, 0U, OptixTraversableHandle{ 10U }, 1U, 2U };
    const OptixInstance four{ {}, 0U, 0U, 0U, 0U, OptixTraversableHandle{ 10U }, 1U, 2U };

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
    const OptixPipelineCompileOptions two{ 1 };
    const OptixPipelineCompileOptions three{ 1, 1U };
    const OptixPipelineCompileOptions four{ 1, 1U, 1 };
    const OptixPipelineCompileOptions five{ 1, 1U, 1, 1 };
    const OptixPipelineCompileOptions six{ 1, 1U, 1, 1, 1U };
    const OptixPipelineCompileOptions seven{ 1, 1U, 1, 1, 1U, "params" };
    const OptixPipelineCompileOptions eight{ 1, 1U, 1, 1, 1U, "params", 1U };
#if OPTIX_VERSION >= 70600
    const OptixPipelineCompileOptions nine{ 1, 1U, 1, 1, 1U, "params", 1U, 1 };
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
    const OptixPipelineCompileOptions one{ 1, 1U, 1, 1, 1U, "params" };
    const OptixPipelineCompileOptions two{ 1, 1U, 1, 1, 1U, "params" };

    EXPECT_NE( one.pipelineLaunchParamsVariableName, two.pipelineLaunchParamsVariableName );
    EXPECT_STREQ( one.pipelineLaunchParamsVariableName, two.pipelineLaunchParamsVariableName );
    EXPECT_EQ( one, two );
    EXPECT_EQ( two, one );
    EXPECT_FALSE( one != two );
    EXPECT_FALSE( two != one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentMotionBlurAreNotEqual )
{
    const OptixPipelineCompileOptions one{ 0 };
    const OptixPipelineCompileOptions two{ 1 };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentTraversableFlagsAreNotEqual )
{
    const OptixPipelineCompileOptions one{ 0, 1U };
    const OptixPipelineCompileOptions two{ 0, 2U };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentNumPayloadValuesAreNotEqual )
{
    const OptixPipelineCompileOptions one{ 0, 0U, 1 };
    const OptixPipelineCompileOptions two{ 0, 0U, 2 };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentNumAttributeValuesAreNotEqual )
{
    const OptixPipelineCompileOptions one{ 0, 0U, 0, 1 };
    const OptixPipelineCompileOptions two{ 0, 0U, 0, 2 };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentExceptionFlagsAreNotEqual )
{
    const OptixPipelineCompileOptions one{ 0, 0U, 0, 0, 1U };
    const OptixPipelineCompileOptions two{ 0, 0U, 0, 0, 2U };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentParamNamesAreNotEqual )
{
    const OptixPipelineCompileOptions one{ 0, 0U, 0, 0, 0U, "param1" };
    const OptixPipelineCompileOptions two{ 0, 0U, 0, 0, 0U, "param2" };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

TEST( TestCompareOptixPipelineCompileOptions, differentPrimitiveTypeFlagsAreNotEqual )
{
    const OptixPipelineCompileOptions one{ 0, 0U, 0, 0, 0U, "param", 1U };
    const OptixPipelineCompileOptions two{ 0, 0U, 0, 0, 0U, "param", 2U };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}

#if OPTIX_VERSION >= 70600
TEST( TestCompareOptixPipelineCompileOptions, differentOpacityMicromapsAreNotEqual )
{
    const OptixPipelineCompileOptions one{ 0, 0U, 0, 0, 0U, "param", 0U, 0 };
    const OptixPipelineCompileOptions two{ 0, 0U, 0, 0, 0U, "param", 0U, 1 };

    EXPECT_NE( one, two );
    EXPECT_NE( two, one );
    EXPECT_FALSE( one == two );
    EXPECT_FALSE( two == one );
}
#endif
