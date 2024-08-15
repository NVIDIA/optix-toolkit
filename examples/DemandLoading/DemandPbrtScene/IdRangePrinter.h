// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <iostream>
#include <vector>

namespace demandPbrtScene {

struct IdRange
{
    IdRange( const std::vector<unsigned int>& ids )
        : m_ids( ids )
    {
    }

    const std::vector<unsigned int>& m_ids;
};

inline std::ostream& operator<<( std::ostream& str, const IdRange& ids )
{
    bool         first{ true };
    unsigned int lastId{};
    bool         rangeOpen{};
    unsigned int rangeCount{};
    auto         concludeRange = [&] {
        if( rangeOpen )
        {
            ++rangeCount;
            str << "-" << lastId;
            if( rangeCount > 5 )
            {
                str << " (" << rangeCount << ')';
            }
        }
    };
    for( unsigned int id : ids.m_ids )
    {
        if( first )
        {
            str << id;
        }
        else if( id == lastId + 1 )
        {
            if (rangeOpen)
            {
                ++rangeCount;
            }
            else
            {
                rangeOpen = true;
                rangeCount = 1;
            }
        }
        else if( rangeOpen )
        {
            concludeRange();
            str << ", " << id;
            rangeOpen = false;
        }
        else
        {
            str << ", " << id;
            rangeOpen = false;
        }
        lastId = id;
        first = false;
    }
    concludeRange();
    return str;
}

}  // namespace demandPbrtScene
