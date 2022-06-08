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

#pragma once

#include <string>
#include <vector>

namespace demandLoading {

class DeviceSet
{
  public:
    /// A single device
    typedef int position;

    /// Simple iteration over set bits.
    struct const_iterator
    {
      public:
        unsigned int operator*() const;
        const_iterator& operator++();
        unsigned int operator++( int );
        bool operator!=( const const_iterator& b ) const;
        bool operator==( const const_iterator& b ) const;

      private:
        friend class DeviceSet;
        const_iterator( const DeviceSet* parent, unsigned int pos );
        const DeviceSet* parent;
        unsigned int     pos;
    };

    /// Default constructor.
    DeviceSet()
        : m_devices( 0 )
    {
    }

    /// Construct singleton set.
    DeviceSet( unsigned int deviceIndex )
        : m_devices( 1 << deviceIndex )
    {
    }

    /// Construct set from vector of device indices.
    DeviceSet( const std::vector<unsigned int>& allDeviceListIndices );

    /// Union
    DeviceSet operator|( const DeviceSet& b ) const
    {
        DeviceSet d = *this;
        d |= b;
        return d;
    }

    /// Assign union
    DeviceSet& operator|=( const DeviceSet& b )
    {
        m_devices |= b.m_devices;
        return *this;
    }

    /// Intersection
    DeviceSet operator&( const DeviceSet& b ) const
    {
        DeviceSet d = *this;
        d &= b;
        return d;
    }

    /// Assign intersection
    DeviceSet& operator&=( const DeviceSet& b )
    {
        m_devices &= b.m_devices;
        return *this;
    }

    /// Difference
    DeviceSet operator-( const DeviceSet& b ) const
    {
        DeviceSet d = *this;
        d -= b;
        return d;
    }

    /// Assign difference.
    DeviceSet& operator-=( const DeviceSet& b )
    {
        m_devices &= ~b.m_devices;
        return *this;
    }

    /// Equality
    bool operator==( const DeviceSet& b ) const { return m_devices == b.m_devices; }

    /// Inequality
    bool operator!=( const DeviceSet& b ) const { return m_devices != b.m_devices; }

    /// Complement
    DeviceSet operator~() const
    {
        DeviceSet r;
        r.m_devices = ~m_devices;
        return r;
    }

    /// Return the device index of the Nth set device.
    unsigned int operator[]( unsigned int n ) const;

    /// Returns the position in the range of [0,count()) based on the all-device-index
    unsigned int getArrayPosition( unsigned int deviceIndex ) const;

    /// Returns true if set is empty.
    bool empty() const { return m_devices == 0; }

    /// Returns number of devices in this set.
    unsigned int count() const;

    /// Returns true if the specified device is a member of this set.
    bool isSet( unsigned int deviceIndex ) const
    {
        const unsigned int mask = 1 << deviceIndex;
        return ( m_devices & mask ) != 0;
    }

    /// Make this set empty.
    void clear() { m_devices = 0; }


    const_iterator begin() const;
    const_iterator end() const;


  private:
    unsigned int m_devices;
};

/// Human readable string in the form {empty}, {all} or {0,1,6}
std::string toString( const DeviceSet& deviceSet );


}  // namespace demandLoading
