// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda_runtime.h>

#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Gui/Camera.h>
#include <OptiXToolkit/Gui/GLDisplay.h>
#include <OptiXToolkit/Gui/glfw3.h>

#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSources/MultiCheckerImage.h>

#include "LaunchParams.h"
#include "PerDeviceOptixState.h"

namespace demandTextureApp
{

class DemandTextureApp
{
  public:
    DemandTextureApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop );
    virtual ~DemandTextureApp() {}

    // Public functions to initialize the app and start rendering
    void setNumLaunches( int numLaunches ) { m_minLaunches = numLaunches; }
    void sceneIsTriangles( bool t ) { m_scene_is_triangles = t; }
    void initDemandLoading();
    virtual void createTexture() = 0;
    void initOptixPipelines( const char* moduleCode, const size_t moduleCodeSize );
    void startLaunchLoop();
    void cleanup();
    void printDemandLoadingStats();
    void resetAccumulator();
    void setMipScale( float scale ) { m_mipScale = scale; }
    void useSparseTextures( bool useSparseTextures ) { m_useSparseTextures = useSparseTextures; }
    void useCascadingTextureSizes( bool useCascade ) { m_useCascadingTextureSizes = useCascade; }
    void setMaxSubframes( int maxSubframes ) { m_maxSubframes = maxSubframes; }

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
    virtual void buildAccel( PerDeviceOptixState& state );
    void createModule( PerDeviceOptixState& state, const char* moduleCode, size_t codeSize );
    void createProgramGroups( PerDeviceOptixState& state );
    void createPipeline( PerDeviceOptixState& state );
    virtual void createSBT( PerDeviceOptixState& state );
    virtual void cleanupState( PerDeviceOptixState& state );

    // Demand loading and texturing system
    demandLoading::TextureDescriptor makeTextureDescriptor( CUaddress_mode addressMode, FilterMode filterMode );
    std::shared_ptr<imageSource::ImageSource> createExrImage( const std::string& filePath );
    
    // OptiX launches
    virtual void initView();
    void setView( float3 eye, float3 lookAt, float3 up, float fovY );
    virtual void initLaunchParams( PerDeviceOptixState& state, unsigned int numDevices );
    unsigned int performLaunches();

    // Displaying and saving images
    virtual void drawGui();
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
    std::string                                    m_outputFileName;
    unsigned int                                   m_render_mode = 0;

    // OptiX states for each device
    std::vector<PerDeviceOptixState> m_perDeviceOptixStates;
    bool m_scene_is_triangles = false;

    // Demand loading and textures
    std::vector<unsigned int> m_textureIds;
    int                       m_launchCycles = 0;
    int                       m_subframeId = 0;
    int                       m_numFilledRequests = 0;
    int                       m_minLaunches = 2;
    bool                      m_useSparseTextures = true;
    bool                      m_useCascadingTextureSizes = false;

    // Number of subframes to do
    int m_maxSubframes = 1000000;

    // Camera and view
    otk::Camera m_camera;
    Projection m_projection = ORTHOGRAPHIC;
    float m_lens_width = 0.0f;
    float4 m_backgroundColor = float4{0.1f, 0.1f, 0.5f, 0.0f};
    float m_mipScale = 1.0f;
    int m_reset_subframe_threshold = 10;

    // Mouse state
    static const int NO_BUTTON = -1;

    double m_mousePrevX  = 0;
    double m_mousePrevY  = 0;
    int    m_mouseButton = NO_BUTTON;

    void panCamera( float3 pan );
    void zoomCamera( float zoom );
    void rotateCamera( float rot );
};

void setGLFWCallbacks( DemandTextureApp* app );

} // namespace demandTextureApp
