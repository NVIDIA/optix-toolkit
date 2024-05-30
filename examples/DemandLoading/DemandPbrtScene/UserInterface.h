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

#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>

#include <memory>

namespace demandPbrtScene {

struct LookAtParams;
struct Options;
struct PerspectiveCamera;
struct UserInterfaceStatistics;

using OutputBuffer = otk::CUDAOutputBuffer<uchar4>;

class UserInterface
{
  public:
    virtual ~UserInterface() = default;

    virtual void initialize( const LookAtParams& lookAt, const PerspectiveCamera& camera ) = 0;
    virtual void cleanup()    = 0;

    // returns true if launch is needed
    virtual bool beforeLaunch( OutputBuffer& output ) = 0;
    virtual void afterLaunch( OutputBuffer& output )  = 0;
    virtual bool shouldClose() const                  = 0;

    virtual void setStatistics(const UserInterfaceStatistics &stats) = 0;
};

std::shared_ptr<UserInterface> createUserInterface( Options& options, RendererPtr renderer, ScenePtr scene );

}  // namespace demandPbrtScene