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

#include <DemandLoading/DeviceContext.h>  // for PageMapping
#include <DemandLoading/Options.h>

namespace demandLoading {

struct PageMappingsContext
{
    PageMapping* filledPages;
    unsigned int numFilledPages;
    unsigned int maxFilledPages;

    unsigned int* invalidatedPages;
    unsigned int  numInvalidatedPages;
    unsigned int  maxInvalidatedPages;

    static void reserve( BulkPinnedMemory* memory, const Options& options )
    {
        memory->reserve<PageMapping>( options.maxFilledPages );
        memory->reserve<unsigned int>( options.maxInvalidatedPages );
    }

    void allocate( BulkPinnedMemory* memory, const Options& options )
    {
        filledPages        = memory->allocate<PageMapping>( options.maxFilledPages );
        numFilledPages     = 0;
        maxFilledPages     = options.maxFilledPages;

        invalidatedPages    = memory->allocate<unsigned int>( options.maxInvalidatedPages );
        numInvalidatedPages = 0;
        maxInvalidatedPages = options.maxInvalidatedPages;
    }

    void clear()
    {
        numFilledPages      = 0;
        numInvalidatedPages = 0;
    }
};

}  // namespace demandLoading
