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

#include <OptiXToolkit/Gui/glad.h>  // Glad insists on being included first.

#include "CudaContext.h"
#include "Dependencies.h"
#include "Options.h"
#include "Params.h"
#include "Sample.h"
#include "Statistics.h"

#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Memory/SyncVector.h>

#include <iostream>
#include <memory>
#include <string>

struct GLFWwindow;

namespace otk {

class GLDisplay;

namespace pbrt {

class Logger;
class SceneLoader;

}  // namespace pbrt
}  // namespace otk

namespace demandPbrtScene {

class Application : public otk::Sample
{
  public:
    Application( int argc, char* argv[] );
    ~Application() override = default;

    void initialize() override;
    void run() override;
    void cleanup() override;

  private:
    unsigned int getCudaDeviceIndex();
    void         launch( otk::CUDAOutputBuffer<uchar4>& outputBuffer );
    void         updateStats( const UserInterfacePtr& ui );
    void         runInteractive();
    void         saveResults( otk::CUDAOutputBuffer<uchar4>& outputBuffer );
    void         runToFile();

    Options    m_options;
    Statistics m_stats{};

    // Dependencies
    CudaContext           m_cuda;
    LoggerPtr             m_logger;
    MeshInfoReaderPtr     m_infoReader;
    PbrtSceneLoaderPtr    m_pbrt;
    DemandLoaderPtr       m_demandLoader;
    GeometryLoaderPtr     m_geometryLoader;
    MaterialLoaderPtr     m_materialLoader;
    GeometryCachePtr      m_geometryCache;
    ImageSourceFactoryPtr m_imageSourceFactory;
    ProxyFactoryPtr       m_proxyFactory;
    RendererPtr           m_renderer;
    DemandTextureCachePtr m_demandTextureCache;
    ProgramGroupsPtr      m_programGroups;
    MaterialResolverPtr   m_materialResolver;
    ScenePtr              m_scene;
    bool                  m_sceneUpdated{};
    Statistics            m_statistics;
};

}  // namespace demandPbrtScene
