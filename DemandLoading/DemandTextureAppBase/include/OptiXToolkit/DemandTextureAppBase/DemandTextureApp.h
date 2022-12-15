//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime.h>

#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Gui/GLDisplay.h>

#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/MultiCheckerImage.h>

#ifdef OPTIX_SAMPLE_USE_CORE_EXR
#include <OptiXToolkit/ImageSource/CoreEXRReader.h>
#define EXRREADER CoreEXRReader
#else 
#include <OptiXToolkit/ImageSource/EXRReader.h>
#define EXRREADER EXRReader
#endif

#include "LaunchParams.h"
#include "PerDeviceOptixState.h"

#include <GLFW/glfw3.h>

namespace demandTextureApp
{

class DemandTextureApp
{
  public:
    DemandTextureApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop );
    virtual ~DemandTextureApp();

    // Public functions to initialize the app and start rendering
    void initDemandLoading();
    virtual void createTexture() = 0;
    void initOptixPipelines( const char* moduleCode );
    void startLaunchLoop();
    void printDemandLoadingStats();

    // GLFW callbacks
    virtual void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods );
    virtual void cursorPosCallback( GLFWwindow* window, double xpos, double ypos );
    virtual void windowSizeCallback( GLFWwindow* window, int width, int height );
    virtual void pollKeys();
    virtual void keyCallback( GLFWwindow* window, int key, int scancode, int action, int mods );
    GLFWwindow* getWindow() { return m_window; }

  protected:
    // OptiX setup
    void createContext( PerDeviceOptixState& state );
    void buildAccel( PerDeviceOptixState& state );
    void createModule( PerDeviceOptixState& state, const char* moduleCode, size_t codeSize );
    void createProgramGroups( PerDeviceOptixState& state );
    void createPipeline( PerDeviceOptixState& state );
    void createSBT( PerDeviceOptixState& state );
    void cleanupState( PerDeviceOptixState& state );

    // Demand loading and texturing system
    demandLoading::TextureDescriptor makeTextureDescriptor( CUaddress_mode addressMode, CUfilter_mode filterMode );
    imageSource::ImageSource* createExrImage( const char* fileName );
    
    // OptiX launches
    void initView();
    void initLaunchParams( PerDeviceOptixState& state, unsigned int numDevices );
    unsigned int performLaunches();

    // Displaying and saving images
    void displayFrame();
    void saveImage();
    bool isInteractive() const { return m_outputFileName.empty(); }

  protected:
    // Window and output buffer
    GLFWwindow*                                    m_window = nullptr;
    std::unique_ptr<otk::CUDAOutputBuffer<uchar4>> m_outputBuffer;
    std::unique_ptr<otk::GLDisplay>                m_glDisplay;
    int                                            m_windowWidth;
    int                                            m_windowHeight;
    std::string                                    m_outputFileName = "";

    // OptiX states for each device
    std::vector<PerDeviceOptixState> m_perDeviceOptixStates;

    // Demand loading and textures
    std::shared_ptr<demandLoading::DemandLoader> m_demandLoader;
    std::vector<unsigned int>                    m_textureIds;
    int                                          m_launchCycles = 0;
    int                                          m_numFilledRequests = 0;

    // Viewpoint description
    const float3 INITIAL_LOOK_FROM{0.5f, 0.5f, 1.0f};
    const float INITIAL_VIEW_DIM = 1.0f;

    float3 m_eye             = INITIAL_LOOK_FROM; 
    float2 m_viewDims        = float2{INITIAL_VIEW_DIM, INITIAL_VIEW_DIM};
    float4 m_backgroundColor = float4{0.1f, 0.1f, 0.5f, 0.0f};

    // Mouse state
    static const int NO_BUTTON = -1;

    double m_mousePrevX  = 0;
    double m_mousePrevY  = 0;
    int    m_mouseButton = NO_BUTTON;
};

void setGLFWCallbacks( DemandTextureApp* app );

} // namespace demandTextureApp