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

namespace demandLoading {

/// Device-side array data, along with its capacity.
template <typename T>
struct DeviceArray
{
    T*           data;
    unsigned int capacity;
};

/// Stale page info.
struct StalePage
{
    unsigned int reserved : 2;
    unsigned int lruVal : 4;
    unsigned int pageId : 26;
};

/// PageMappings are used to update page table entries.
struct PageMapping
{
    unsigned int       id;      // page table index
    unsigned int       lruVal;  // least-recently used counter
    unsigned long long page;    // page table entry
};

/// Device-side demand texture context.
struct DeviceContext
{
    DeviceArray<unsigned long long> pageTable;
    unsigned int                    maxNumPages;  // the actual capacity of the page table is smaller
    unsigned int*                   referenceBits;
    unsigned int*                   residenceBits;
    unsigned int*                   lruTable;
    DeviceArray<unsigned int>       requestedPages;
    DeviceArray<StalePage>          stalePages;
    DeviceArray<unsigned int>       evictablePages;
    DeviceArray<unsigned int>       arrayLengths;  // 0=requestedPages, 1=stalePages, 2=evictablePages
    DeviceArray<PageMapping>        filledPages;
    DeviceArray<unsigned int>       invalidatedPages;
    bool                            requestIfResident; 
    unsigned int                    poolIndex;  // Needed when returning copied context to pool.
};

/// Interpretation of indices in DeviceContext::arrayLengths.
enum ArrayLengthsIndex
{
    PAGE_REQUESTS_LENGTH   = 0,
    STALE_PAGES_LENGTH     = 1,
    EVICTABLE_PAGES_LENGTH = 2
};

}  // namespace demandLoading
