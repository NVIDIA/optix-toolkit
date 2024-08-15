// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
    unsigned int                    maxTextures;
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
    EVICTABLE_PAGES_LENGTH = 2,
    NUM_ARRAY_LENGTHS
};

}  // namespace demandLoading
