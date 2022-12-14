//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "Util/DeviceSet.h"
#include "Util/Exception.h"

#include <algorithm>
#include <sstream>

namespace {

// Return the 1-based index of the least significant bit that's set.
inline unsigned int leastSignificantBitSet( unsigned int x )
{
    if( !x )
        return 0;
    unsigned int bit = 33U;
    do
    {
        x <<= 1;
        bit--;
    } while( x );
    return bit;
}

// Return the number of bits set in a
inline unsigned int popCount( unsigned int a )
{
    unsigned int c;
    c = ( a & 0x55555555 ) + ( ( a >> 1 ) & 0x55555555 );
    c = ( c & 0x33333333 ) + ( ( c >> 2 ) & 0x33333333 );
    c = ( c & 0x0f0f0f0f ) + ( ( c >> 4 ) & 0x0f0f0f0f );
    c = ( c & 0x00ff00ff ) + ( ( c >> 8 ) & 0x00ff00ff );
    c = ( c & 0x0000ffff ) + ( c >> 16 );
    return c;
}
}

namespace demandLoading {

DeviceSet::DeviceSet( const std::vector<unsigned int>& allDeviceListIndices )
    : m_devices( 0 )
{
    for( const unsigned int deviceIndex : allDeviceListIndices )
    {
        m_devices |= ( 1 << deviceIndex );
    }
}

unsigned int DeviceSet::count() const
{
    // Table of counts for up to 5 devices
    // clang-format off
    static const unsigned int deviceCounts[] = {
        0, // 0
        1, // 1
        1, 2, // 10, 11
        1, 2, 2, 3, // 100, 101, 110, 111
        1, 2, 2, 3, 2, 3, 3, 4, // 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5
    };
    // clang-format on

    return ( m_devices < 32 ) ? deviceCounts[m_devices] : popCount( m_devices );
}

std::string toString( const DeviceSet& devices )
{
    if( devices.empty() )
        return "{empty}";

    std::ostringstream out;
    out << "{";
    for( DeviceSet::const_iterator iter = devices.begin(); iter != devices.end(); ++iter )
    {
        if( iter != devices.begin() )
            out << ",";
        out << *iter;
    }
    out << "}";
    return out.str();
}

unsigned int DeviceSet::const_iterator::operator*() const
{
    return pos;
}

DeviceSet::const_iterator& DeviceSet::const_iterator::operator++()
{
    if( pos + 1 == sizeof( parent->m_devices ) * 8 )
    {
        pos = -1;
    }
    else
    {
        unsigned int mask = ( 1 << ( pos + 1 ) ) - 1;
        pos               = leastSignificantBitSet( parent->m_devices & ~mask ) - 1;
    }
    return *this;
}

unsigned int DeviceSet::const_iterator::operator++( int )
{
    unsigned int temp = **this;
    ++*this;
    return temp;
}


bool DeviceSet::const_iterator::operator!=( const const_iterator& b ) const
{
    return !( ( *this ) == b );
}

bool DeviceSet::const_iterator::operator==( const const_iterator& b ) const
{
    return parent == b.parent && pos == b.pos;
}

unsigned int DeviceSet::operator[]( unsigned int n ) const
{
    const_iterator it = begin();
    while( n-- )
        ++it;
    return *it;
}

unsigned int DeviceSet::getArrayPosition( unsigned int deviceIndex ) const
{
    DEMAND_ASSERT_MSG( isSet( deviceIndex ), "expected deviceIndex to be set in DeviceSet" );
    const unsigned int mask = ( 1u << deviceIndex ) - 1u;
    return popCount( m_devices & mask );
}

DeviceSet::const_iterator::const_iterator( const DeviceSet* parent, unsigned pos )
    : parent( parent )
    , pos( pos )
{
}

DeviceSet::const_iterator DeviceSet::begin() const
{
    return const_iterator( this, leastSignificantBitSet( m_devices ) - 1 );
}

DeviceSet::const_iterator DeviceSet::end() const
{
    return const_iterator( this, -1 );
}

}  // namespace demandLoading
