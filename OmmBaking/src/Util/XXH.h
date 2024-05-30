// xxHash Library
//  Copyright( c ) 2012 - 2021 Yann Collet
// All rights reserved.
//
//  BSD 2 - Clause License( https://www.opensource.org/licenses/bsd-license.php)
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met :
//
//  * Redistributions of source code must retain the above copyright notice, this
//  list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above copyright notice, this
//  list of conditions and the following disclaimer in the documentation and /or
// other materials provided with the distribution.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
//  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION ) HOWEVER CAUSED AND ON
//  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  ( INCLUDING NEGLIGENCE OR OTHERWISE ) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

__device__ unsigned int XXH( const uint4& p )
{
    constexpr unsigned int PRIME32_2 = 2246822519u, PRIME32_3 = 3266489917u;
    constexpr unsigned int PRIME32_4 = 668265263u, PRIME32_5 = 374761393u;
    unsigned int           h32 = p.w + PRIME32_5 + p.x * PRIME32_3;
    h32                        = PRIME32_4 * ( ( h32 << 17 ) | ( h32 >> ( 32 - 17 ) ) );
    h32 += p.y * PRIME32_3;
    h32 = PRIME32_4 * ( ( h32 << 17 ) | ( h32 >> ( 32 - 17 ) ) );
    h32 += p.z * PRIME32_3;
    h32 = PRIME32_4 * ( ( h32 << 17 ) | ( h32 >> ( 32 - 17 ) ) );
    h32 = PRIME32_2 * ( h32 ^ ( h32 >> 15 ) );
    h32 = PRIME32_3 * ( h32 ^ ( h32 >> 13 ) );
    return h32 ^ ( h32 >> 16 );
}

__device__ unsigned int XXH( const uint4& p0, const uint4& p1 )
{
    constexpr unsigned int PRIME32_2 = 2246822519u, PRIME32_3 = 3266489917u;
    constexpr unsigned int PRIME32_4 = 668265263u, PRIME32_5 = 374761393u;
    unsigned int           h32 = p1.w + PRIME32_5 + p0.x * PRIME32_3;
    h32                        = PRIME32_4 * ( ( h32 << 17 ) | ( h32 >> ( 32 - 17 ) ) );
    h32 += p0.y * PRIME32_3;
    h32 = PRIME32_4 * ( ( h32 << 17 ) | ( h32 >> ( 32 - 17 ) ) );
    h32 += p0.z * PRIME32_3;
    h32 = PRIME32_4 * ( ( h32 << 17 ) | ( h32 >> ( 32 - 17 ) ) );
    h32 += p0.w * PRIME32_3;
    h32 = PRIME32_4 * ( ( h32 << 17 ) | ( h32 >> ( 32 - 17 ) ) );
    h32 += p1.x * PRIME32_3;
    h32 = PRIME32_4 * ( ( h32 << 17 ) | ( h32 >> ( 32 - 17 ) ) );
    h32 += p1.y * PRIME32_3;
    h32 = PRIME32_4 * ( ( h32 << 17 ) | ( h32 >> ( 32 - 17 ) ) );
    h32 += p1.z * PRIME32_3;
    h32 = PRIME32_4 * ( ( h32 << 17 ) | ( h32 >> ( 32 - 17 ) ) );
    h32 = PRIME32_2 * ( h32 ^ ( h32 >> 15 ) );
    h32 = PRIME32_3 * ( h32 ^ ( h32 >> 13 ) );
    return h32 ^ ( h32 >> 16 );
}
