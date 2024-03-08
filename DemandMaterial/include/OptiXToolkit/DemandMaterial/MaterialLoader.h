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

#include <memory>
#include <vector>

namespace demandLoading {
class DemandLoader;
}  // namespace demandLoading

namespace demandMaterial {

using uint_t = unsigned int;

class MaterialLoader
{
  public:
    virtual ~MaterialLoader() = default;

    virtual const char* getCHFunctionName() const = 0;

    virtual uint_t add()               = 0;
    virtual void   remove( uint_t id ) = 0;

    virtual std::vector<uint_t> requestedMaterialIds() const = 0;

    /// Return whether or not proxy ids are recycled as they are removed.
    virtual bool getRecycleProxyIds() const = 0;

    /// Enable or disable whether or not proxy ids are recycled as they are removed.
    /// The default is to not recycle proxy ids.
    virtual void setRecycleProxyIds( bool enable ) = 0;
};

std::shared_ptr<MaterialLoader> createMaterialLoader( demandLoading::DemandLoader* demandLoader );

}  // namespace demandMaterial
