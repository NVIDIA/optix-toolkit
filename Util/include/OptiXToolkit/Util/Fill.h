//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "SyncVector.h"

#include <algorithm>
#include <vector>

namespace otk {

/// Fill a fixed-length array of Ts with a value of type U that can be converted to T.
template <typename T, size_t N, typename U>
void fill( T ( &ary )[N], U value )
{
    std::fill( std::begin( ary ), std::end( ary ), static_cast<T>( value ) );
}

/// Fill a SyncVector<T> with a value of type U that can be converted to T.
template <typename T, typename U>
void fill( SyncVector<T>& vec, U value )
{
    std::fill( std::begin( vec ), std::end( vec ), static_cast<T>( value ) );
}

/// Fill a std::vector<T> with a value of type U that can be converted to T.
template <typename T, typename U>
void fill( std::vector<T>& vec, U value )
{
    std::fill( std::begin( vec ), std::end( vec ), static_cast<T>( value ) );
}

}  // namespace otk
