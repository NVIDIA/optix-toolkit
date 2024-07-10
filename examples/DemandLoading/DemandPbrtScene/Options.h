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

#include <functional>
#include <string>

#include <OptiXToolkit/ShaderUtil/vec_math.h>

namespace demandPbrtScene {

struct Options
{
    std::string program;
    std::string sceneFile;
    std::string outFile;
    int         width{ 768 };
    int         height{ 512 };
    float3      background{};
    int         warmupFrames{ 0 };
    bool        oneShotGeometry{};
    bool        oneShotMaterial{};
    bool        verboseLoading{};
    bool        verboseProxyGeometryResolution{};
    bool        verboseProxyMaterialResolution{};
    bool        verboseSceneDecomposition{};
    bool        verboseTextureCreation{};
    bool        sortProxies{};
    bool        sync{};
    bool        usePinholeCamera{ true };
    bool        faceForward{};
    bool        debug{};
    bool        oneShotDebug{};
    int2        debugPixel{};
    int         renderMode;
};

using UsageFn = void( const char* program, const char* message );
Options parseOptions( int argc, char* argv[], const std::function<UsageFn>& usage );
Options parseOptions( int argc, char* argv[] );

}  // namespace demandPbrtScene
