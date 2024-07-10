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

#include "Dependencies.h"
#include "Traversable.h"

#include <optix.h>

#include <cuda.h>

#include <memory>
#include <vector>

namespace otk {

struct DebugLocation;

}  // namespace otk

namespace demandPbrtScene {

struct PerspectiveCamera;
struct Options;
class Scene;

class Renderer
{
  public:
    virtual ~Renderer() = default;

    virtual void initialize( CUstream stream ) = 0;
    virtual void cleanup()                     = 0;

    virtual const otk::DebugLocation&          getDebugLocation() const          = 0;
    virtual const LookAtParams&                getLookAt() const                 = 0;
    virtual const PerspectiveCamera&           getCamera() const                 = 0;
    virtual Params&                            getParams()                       = 0;
    virtual OptixDeviceContext                 getDeviceContext() const          = 0;
    virtual const OptixPipelineCompileOptions& getPipelineCompileOptions() const = 0;

    virtual void setDebugLocation( const otk::DebugLocation& data )              = 0;
    virtual void setCamera( const PerspectiveCamera& definition )                = 0;
    virtual void setLookAt( const LookAtParams& lookAt )                         = 0;
    virtual void setProgramGroups( const std::vector<OptixProgramGroup>& value ) = 0;

    virtual void beforeLaunch( CUstream stream )           = 0;
    virtual void launch( CUstream stream, uchar4* pixels ) = 0;
    virtual void afterLaunch()                             = 0;
    virtual void fireOneDebugDump()                        = 0;
    virtual void setClearAccumulator()                     = 0;
};

RendererPtr createRenderer( const Options& options, int numAttributes );

}  // namespace demandPbrtScene
