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

#include <OptiXToolkit/Memory/SyncVector.h>

#include <mutex>
#include <vector>

namespace demandLoading {
class DemandLoader;
}  // namespace demandLoading

namespace demandGeometryViewer {

using uint_t = unsigned int;

class DemandMaterial
{
  public:
    DemandMaterial( demandLoading::DemandLoader* loader );

    uint_t add();
    void remove( uint_t id );

    std::vector<uint_t> requestedMaterialIds() const { return m_requestedMaterials; }

  private:
    demandLoading::DemandLoader* m_loader;
    std::vector<uint_t>          m_materialIds;
    std::vector<uint_t>          m_requestedMaterials;
    std::mutex                   m_requestedMaterialsMutex;

    bool loadMaterial( CUstream stream, uint_t pageId, void** pageTableEntry );

    static bool callback( CUstream stream, uint_t pageIndex, void* context, void** pageTableEntry )
    {
        return static_cast<DemandMaterial*>( context )->loadMaterial( stream, pageIndex, pageTableEntry );
    }
};

}  // namespace demandGeometryViewer
