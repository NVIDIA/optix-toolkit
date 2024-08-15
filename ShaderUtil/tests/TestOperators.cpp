// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <gtest/gtest.h>

using namespace testing;

// clang-format off
template <typename T> struct ComponentType      { using Type = void; };
template <> struct ComponentType<short2>        { using Type = short; };
template <> struct ComponentType<ushort2>       { using Type = unsigned short; };
template <> struct ComponentType<int2>          { using Type = int; };
template <> struct ComponentType<uint2>         { using Type = unsigned int; };
template <> struct ComponentType<long2>         { using Type = long; };
template <> struct ComponentType<ulong2>        { using Type = unsigned long; };
template <> struct ComponentType<longlong2>     { using Type = long long; };
template <> struct ComponentType<ulonglong2>    { using Type = unsigned long long; };
template <> struct ComponentType<float2>        { using Type = float; };
template <> struct ComponentType<double2>       { using Type = double; };
template <> struct ComponentType<short3>        { using Type = short; };
template <> struct ComponentType<ushort3>       { using Type = unsigned short; };
template <> struct ComponentType<int3>          { using Type = int; };
template <> struct ComponentType<uint3>         { using Type = unsigned int; };
template <> struct ComponentType<long3>         { using Type = long; };
template <> struct ComponentType<ulong3>        { using Type = unsigned long; };
template <> struct ComponentType<longlong3>     { using Type = long long; };
template <> struct ComponentType<ulonglong3>    { using Type = unsigned long long; };
template <> struct ComponentType<float3>        { using Type = float; };
template <> struct ComponentType<double3>       { using Type = double; };
template <> struct ComponentType<short4>        { using Type = short; };
template <> struct ComponentType<ushort4>       { using Type = unsigned short; };
template <> struct ComponentType<int4>          { using Type = int; };
template <> struct ComponentType<uint4>         { using Type = unsigned int; };
template <> struct ComponentType<long4>         { using Type = long; };
template <> struct ComponentType<ulong4>        { using Type = unsigned long; };
template <> struct ComponentType<longlong4>     { using Type = long long; };
template <> struct ComponentType<ulonglong4>    { using Type = unsigned long long; };
template <> struct ComponentType<float4>        { using Type = float; };
template <> struct ComponentType<double4>       { using Type = double; };

template <typename T> T createOne();
template <> short               createOne() { return static_cast<short>( 1 ); }
template <> unsigned short      createOne() { return static_cast<unsigned short>( 1 ); }
template <> int                 createOne() { return 1; }
template <> unsigned int        createOne() { return 1U; }
template <> long                createOne() { return 1L; }
template <> unsigned long       createOne() { return 1UL; }
template <> long long           createOne() { return 1LL; }
template <> unsigned long long  createOne() { return 1ULL; }
template <> float               createOne() { return 1.0f; }
template <> double              createOne() { return 1.0; }

template <typename T> T createTwo();
template <> short               createTwo() { return static_cast<short>( 2 ); }
template <> unsigned short      createTwo() { return static_cast<unsigned short>( 2 ); }
template <> int                 createTwo() { return 2; }
template <> unsigned int        createTwo() { return 2U; }
template <> long                createTwo() { return 2L; }
template <> unsigned long       createTwo() { return 2UL; }
template <> long long           createTwo() { return 2LL; }
template <> unsigned long long  createTwo() { return 2ULL; }
template <> float               createTwo() { return 2.0f; }
template <> double              createTwo() { return 2.0; }

template <typename T> T createThree();
template <> short               createThree() { return static_cast<short>( 3 ); }
template <> unsigned short      createThree() { return static_cast<unsigned short>( 3 ); }
template <> int                 createThree() { return 3; }
template <> unsigned int        createThree() { return 3U; }
template <> long                createThree() { return 3L; }
template <> unsigned long       createThree() { return 3UL; }
template <> long long           createThree() { return 3LL; }
template <> unsigned long long  createThree() { return 3ULL; }
template <> float               createThree() { return 3.0f; }
template <> double              createThree() { return 3.0; }
// clang-format on

template <typename T>
class Vector2Test : public Test
{
  protected:
    using C = typename ComponentType<T>::Type;
    C one{ createOne<C>() };
    C two{ createTwo<C>() };
    C three{ createThree<C>() };
    T oneOne{ this->one, this->one };
    T oneTwo{ this->one, this->two };
    T twoOne{ this->two, this->one };
    T twoTwo{ this->two, this->two };
    T threeThree{ this->three, this->three };
};

using Vector2Types = Types<short2, ushort2, int2, long2, longlong2, uint2, ulong2, ulonglong2, float2, double2>;
TYPED_TEST_SUITE( Vector2Test, Vector2Types );

TYPED_TEST( Vector2Test, equal )
{
    using T = TypeParam;
    T oneTwoCopy{ this->oneTwo };

    ASSERT_EQ( this->oneTwo, oneTwoCopy );
}

TYPED_TEST( Vector2Test, notEqual )
{
    using T = TypeParam;

    ASSERT_NE( this->oneOne, this->oneTwo );
    ASSERT_NE( this->oneOne, this->twoOne );
    ASSERT_NE( this->twoOne, this->oneTwo );
}

TYPED_TEST( Vector2Test, add )
{
    using T = TypeParam;
    T val   = this->oneOne;

    ( val += this->one ) += this->oneOne;

    ASSERT_EQ( this->threeThree, this->oneTwo + this->twoOne );
    ASSERT_EQ( this->twoTwo, this->oneOne + this->one );
    ASSERT_EQ( this->twoTwo, this->one + this->oneOne );
    ASSERT_EQ( this->threeThree, val );
}

TYPED_TEST( Vector2Test, subtract )
{
    using T = TypeParam;
    T val   = this->threeThree;

    ( val -= this->one ) -= this->oneOne;

    ASSERT_EQ( this->oneTwo, this->threeThree - this->twoOne );
    ASSERT_EQ( this->oneOne, this->twoTwo - this->one );
    ASSERT_EQ( this->oneOne, this->two - this->oneOne );
    ASSERT_EQ( val, this->oneOne );
}

TYPED_TEST( Vector2Test, multiply )
{
    using T          = TypeParam;
    const T fourFour = this->twoTwo + this->twoTwo;
    T       val      = this->oneOne;

    ( val *= this->two ) *= this->twoTwo;

    ASSERT_EQ( fourFour, this->twoTwo * this->twoTwo );
    ASSERT_EQ( this->twoTwo, this->oneOne * this->two );
    ASSERT_EQ( this->twoTwo, this->two * this->oneOne );
    ASSERT_EQ( val, fourFour );
}

TYPED_TEST( Vector2Test, divide )
{
    using T          = TypeParam;
    const T fourFour = this->twoTwo + this->twoTwo;
    T       val      = fourFour;

    ( val /= this->two ) /= this->twoTwo;

    ASSERT_EQ( this->twoTwo, fourFour / this->twoTwo );
    ASSERT_EQ( this->oneOne, this->twoTwo / this->two );
    ASSERT_EQ( this->oneOne, this->two / this->twoTwo );
    ASSERT_EQ( val, this->oneOne );
}

template <typename T>
class SignedVector2Test : public Vector2Test<T>
{
  protected:
    using C = typename ComponentType<T>::Type;
    C minusOne{ static_cast<C>( -this->one ) };
    T minusOneOne{ minusOne, minusOne };
};
using SignedVector2Types = Types<short2, int2, long2, longlong2, float2, double2>;
TYPED_TEST_SUITE( SignedVector2Test, SignedVector2Types );

TYPED_TEST( SignedVector2Test, negate )
{
    ASSERT_EQ( this->minusOneOne, -this->oneOne );
}

template <typename T>
class Vector3Test : public Test
{
  protected:
    using C = typename ComponentType<T>::Type;
    C one{ createOne<C>() };
    C two{ createTwo<C>() };
    C three{ createThree<C>() };
    T oneOneOne{ this->one, this->one, this->one };
    T oneOneTwo{ this->one, this->one, this->two };
    T oneTwoOne{ this->one, this->two, this->one };
    T oneTwoTwo{ this->one, this->two, this->two };
    T twoOneOne{ this->two, this->one, this->one };
    T twoOneTwo{ this->two, this->one, this->two };
    T twoTwoOne{ this->two, this->two, this->one };
    T twoTwoTwo{ this->two, this->two, this->two };
    T threeThreeThree{ this->three, this->three, this->three };
};

using Vector3Types = Types<short3, ushort3, int3, long3, longlong3, uint3, ulong3, ulonglong3, float3, double3>;
TYPED_TEST_SUITE( Vector3Test, Vector3Types );

TYPED_TEST( Vector3Test, equal )
{
    ASSERT_EQ( this->oneOneTwo, this->oneOneTwo );
    ASSERT_EQ( this->oneTwoOne, this->oneTwoOne );
    ASSERT_EQ( this->twoOneOne, this->twoOneOne );
    ASSERT_FALSE( this->oneOneOne == this->oneOneTwo );
    ASSERT_FALSE( this->oneOneOne == this->oneTwoOne );
    ASSERT_FALSE( this->oneOneOne == this->twoOneTwo );
}

TYPED_TEST( Vector3Test, notEqual )
{
    ASSERT_NE( this->oneOneTwo, this->oneOneOne );
    ASSERT_NE( this->oneTwoOne, this->oneOneOne );
    ASSERT_NE( this->twoOneOne, this->oneOneOne );
}

TYPED_TEST( Vector3Test, add )
{
    using T = TypeParam;
    const T threeThreeTwo{ this->three, this->three, this->two };
    T       val = this->oneOneOne;

    ( val += this->one ) += this->oneOneOne;

    ASSERT_EQ( threeThreeTwo, this->oneTwoOne + this->twoOneOne );
    ASSERT_EQ( this->twoTwoTwo, this->oneOneOne + this->one );
    ASSERT_EQ( this->twoTwoTwo, this->one + this->oneOneOne );
    ASSERT_EQ( this->threeThreeThree, val );
}

TYPED_TEST( Vector3Test, subtract )
{
    using T = TypeParam;
    T val   = this->threeThreeThree;

    ( val -= this->one ) -= this->oneOneOne;

    ASSERT_EQ( this->oneTwoTwo, this->threeThreeThree - this->twoOneOne );
    ASSERT_EQ( this->oneOneOne, this->twoTwoTwo - this->one );
    ASSERT_EQ( this->oneOneOne, this->two - this->oneOneOne );
    ASSERT_EQ( val, this->oneOneOne );
}

TYPED_TEST( Vector3Test, multiply )
{
    using T              = TypeParam;
    const T fourFourFour = this->twoTwoTwo + this->twoTwoTwo;
    T       val          = this->oneOneOne;

    ( val *= this->two ) *= this->twoTwoTwo;

    ASSERT_EQ( fourFourFour, this->twoTwoTwo * this->twoTwoTwo );
    ASSERT_EQ( this->twoTwoTwo, this->oneOneOne * this->two );
    ASSERT_EQ( this->twoTwoTwo, this->two * this->oneOneOne );
    ASSERT_EQ( val, fourFourFour );
}

TYPED_TEST( Vector3Test, divide )
{
    using T              = TypeParam;
    const T fourFourFour = this->twoTwoTwo + this->twoTwoTwo;
    T       val          = fourFourFour;

    ( val /= this->two ) /= this->twoTwoTwo;

    ASSERT_EQ( this->twoTwoTwo, fourFourFour / this->twoTwoTwo );
    ASSERT_EQ( this->oneOneOne, this->twoTwoTwo / this->two );
    ASSERT_EQ( this->oneOneOne, this->two / this->twoTwoTwo );
    ASSERT_EQ( val, this->oneOneOne );
}

template <typename T>
class SignedVector3Test : public Vector3Test<T>
{
  protected:
    using C = typename ComponentType<T>::Type;
    C minusOne{ static_cast<C>( -this->one ) };
    T minusOneOneOne{ minusOne, minusOne, minusOne };
};
using SignedVector3Types = Types<short3, int3, long3, longlong3, float3, double3>;
TYPED_TEST_SUITE( SignedVector3Test, SignedVector3Types );

TYPED_TEST( SignedVector3Test, negate )
{
    ASSERT_EQ( this->minusOneOneOne, -this->oneOneOne );
}

template <typename T>
class Vector4Test : public Test
{
  protected:
    using C = typename ComponentType<T>::Type;
    C one{ createOne<C>() };
    C two{ createTwo<C>() };
    C three{ createThree<C>() };
    T oneOneOneOne{ this->one, this->one, this->one, this->one };
    T oneOneOneTwo{ this->one, this->one, this->one, this->two };
    T oneOneTwoOne{ this->one, this->one, this->two, this->one };
    T oneOneTwoTwo{ this->one, this->one, this->two, this->two };
    T oneTwoOneOne{ this->one, this->two, this->one, this->one };
    T oneTwoOneTwo{ this->one, this->two, this->one, this->two };
    T oneTwoTwoOne{ this->one, this->two, this->two, this->one };
    T oneTwoTwoTwo{ this->one, this->two, this->two, this->two };
    T twoOneOneOne{ this->two, this->one, this->one, this->one };
    T twoTwoTwoTwo{ this->two, this->two, this->two, this->two };
    T threeThreeThreeThree{ this->three, this->three, this->three, this->three };
};

using Vector4Types = Types<short4, ushort4, int4, long4, longlong4, uint4, ulong4, ulonglong4, float4, double4>;
TYPED_TEST_SUITE( Vector4Test, Vector4Types );

TYPED_TEST( Vector4Test, equal )
{
    ASSERT_EQ( this->oneOneOneTwo, this->oneOneOneTwo );
    ASSERT_EQ( this->oneOneTwoOne, this->oneOneTwoOne );
    ASSERT_EQ( this->oneTwoOneOne, this->oneTwoOneOne );
    ASSERT_EQ( this->twoOneOneOne, this->twoOneOneOne );
    ASSERT_FALSE( this->oneOneOneOne == this->oneOneOneTwo );
    ASSERT_FALSE( this->oneOneOneOne == this->oneOneTwoOne );
    ASSERT_FALSE( this->oneOneOneOne == this->oneTwoOneOne );
    ASSERT_FALSE( this->oneOneOneOne == this->twoOneOneOne );
}

TYPED_TEST( Vector4Test, notEqual )
{
    ASSERT_NE( this->oneOneOneOne, this->oneOneOneTwo );
    ASSERT_NE( this->oneOneOneOne, this->oneOneTwoOne );
    ASSERT_NE( this->oneOneOneOne, this->oneTwoOneOne );
    ASSERT_NE( this->oneOneOneOne, this->twoOneOneOne );
}

TYPED_TEST( Vector4Test, add )
{
    using T = TypeParam;
    using C = typename ComponentType<T>::Type;
    const T threeThreeThreeThree{ this->three, this->three, this->three, this->three };
    const T zeroZeroZeroOne{ C{}, C{}, C{}, this->one };
    const T zeroZeroOneZero{ C{}, C{}, this->one, C{} };
    const T zeroOneZeroZero{ C{}, this->one, C{}, C{} };
    const T oneZeroZeroZero{ this->one, C{}, C{}, C{} };
    T       val{ this->oneOneOneOne };

    ( val += this->one ) += this->oneOneOneOne;

    ASSERT_EQ( this->twoTwoTwoTwo, this->oneOneOneOne + this->oneOneOneOne );
    ASSERT_EQ( this->oneOneOneTwo, this->oneOneOneOne + zeroZeroZeroOne );
    ASSERT_EQ( this->oneOneTwoOne, this->oneOneOneOne + zeroZeroOneZero );
    ASSERT_EQ( this->oneTwoOneOne, this->oneOneOneOne + zeroOneZeroZero );
    ASSERT_EQ( this->twoOneOneOne, this->oneOneOneOne + oneZeroZeroZero );
    ASSERT_EQ( this->twoTwoTwoTwo, this->oneOneOneOne + this->one );
    ASSERT_EQ( this->twoTwoTwoTwo, this->one + this->oneOneOneOne );
    ASSERT_EQ( threeThreeThreeThree, val );
}

TYPED_TEST( Vector4Test, subtract )
{
    using T = TypeParam;
    T val   = this->threeThreeThreeThree;

    ( val -= this->one ) -= this->oneOneOneOne;

    ASSERT_EQ( this->oneTwoTwoTwo, this->threeThreeThreeThree - this->twoOneOneOne );
    ASSERT_EQ( this->oneOneOneOne, this->twoTwoTwoTwo - this->one );
    ASSERT_EQ( this->oneOneOneOne, this->two - this->oneOneOneOne );
    ASSERT_EQ( val, this->oneOneOneOne );
}

template <typename T>
class SignedVector4Test : public Vector4Test<T>
{
  protected:
    using C = typename ComponentType<T>::Type;
    C minusOne{ static_cast<C>( -this->one ) };
    T minusOneOneOneOne{ minusOne, minusOne, minusOne, minusOne };
};
using SignedVector4Types = Types<short4, int4, long4, longlong4, float4, double4>;
TYPED_TEST_SUITE( SignedVector4Test, SignedVector4Types );

TYPED_TEST( SignedVector4Test, negate )
{
    ASSERT_EQ( this->minusOneOneOneOne, -this->oneOneOneOne );
}
