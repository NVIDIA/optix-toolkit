// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Gui/glad.h>  // Glad insists on being included first.

#include "DemandPbrtScene/CudaContext.h"
#include "DemandPbrtScene/Dependencies.h"
#include "DemandPbrtScene/Options.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/Sample.h"
#include "DemandPbrtScene/Statistics.h"

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
    GeometryResolverPtr   m_geometryResolver;
    ScenePtr              m_scene;
    bool                  m_sceneUpdated{};
    Statistics            m_statistics;
};

}  // namespace demandPbrtScene
