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

#include <OptiXToolkit/OTKAppBase/OTKAppShapeMaker.h>
#include <OptiXToolkit/OTKAppBase/OTKAppLaunchParams.h>
#include <OptiXToolkit/OTKAppBase/OTKAppPerDeviceOptixState.h>

namespace otkApp
{

enum OTKAppUIMode {UI_NONE=0, UI_IMAGEVIEW, UI_FIRSTPERSON};

class OTKApp
{
  public:
    OTKApp( const char* appTitle, unsigned int width, unsigned int height,
            const std::string& outFileName, bool glInterop, OTKAppUIMode uiMode );
    virtual ~OTKApp() {}

    // Public functions to initialize the app and start rendering
    void setNumLaunches( int numLaunches ) { m_minLaunches = numLaunches; }
    void initDemandLoading( demandLoading::Options options );
    void initOptixPipelines( const char* moduleCode, const size_t moduleCodeSize );
    void startLaunchLoop();
    void cleanup();
    void printDemandLoadingStats();
    void resetAccumulator();
    void setMipScale( float scale ) { m_mipScale = scale; }
    void setMaxSubframes( int maxSubframes ) { m_maxSubframes = maxSubframes; }

    SurfaceTexture makeSurfaceTex( int kd, int kdtex, int ks, int kstex, int kt, int kttex, float roughness, float ior );
    void addShapeToScene( std::vector<Vert>& shape, unsigned int materialId );
    void copyGeometryToDevice();

    // Specify the OptiX program names
    void setRaygenProgram( const char* raygenName ) { m_raygenName = raygenName; }
    void setMissProgram( const char* missName ) { m_missName = missName; }
    void setClosestHitProgram( const char* closestHitName ) { m_closestHitName = closestHitName; }
    void setIntersectionProgram( const char* intersectName ) { m_intersectName = intersectName; }
    void setAnyhitProgram( const char* anyhitName ) { m_anyhitName = anyhitName; }
    void setOptixGeometryFlags( unsigned int flags ) { m_optixGeometryFlags = flags; }

    // GLFW callbacks
    virtual void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods );
    virtual void cursorPosCallback( GLFWwindow* window, double xpos, double ypos );
    virtual void windowSizeCallback( GLFWwindow* window, int width, int height );
    virtual void pollKeys();
    virtual void keyCallback( GLFWwindow* window, int key, int scancode, int action, int mods );
    GLFWwindow* getWindow() { return m_window; }

  protected:
    // OptiX setup
    void createContext( OTKAppPerDeviceOptixState& state );
    virtual void buildAccel( OTKAppPerDeviceOptixState& state );
    void createModule( OTKAppPerDeviceOptixState& state, const char* moduleCode, size_t codeSize );
    void createProgramGroups( OTKAppPerDeviceOptixState& state );
    void createPipeline( OTKAppPerDeviceOptixState& state );
    virtual void createSBT( OTKAppPerDeviceOptixState& state );
    virtual void cleanupState( OTKAppPerDeviceOptixState& state );

    // Demand loading and texturing system
    demandLoading::TextureDescriptor makeTextureDescriptor( CUaddress_mode addressMode, FilterMode filterMode );
    std::shared_ptr<imageSource::ImageSource> createExrImage( const std::string& filePath );
    
    // OptiX launches
    virtual void initView();
    void setView( float3 eye, float3 lookAt, float3 up, float fovY );
    virtual void initLaunchParams( OTKAppPerDeviceOptixState& state, unsigned int numDevices );
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
    std::vector<OTKAppPerDeviceOptixState> m_perDeviceOptixStates;

    // Geometry data for scene
    std::vector<float4> m_vertices;
    std::vector<float3> m_normals;
    std::vector<float2> m_tex_coords;
    std::vector<uint32_t> m_material_indices;
    std::vector<OTKAppTriangleHitGroupData> m_materials;

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

    // User Interface
    static const int NO_BUTTON = -1;

    OTKAppUIMode m_uiMode      = UI_NONE;
    double       m_mousePrevX  = 0;
    double       m_mousePrevY  = 0;
    int          m_mouseButton = NO_BUTTON;

    void panCamera( float3 pan );
    void zoomCamera( float zoom );
    void rotateCamera( float rot );

    // OptiX program names
    const char* m_raygenName = "__raygen__rg";
    const char* m_missName = "__miss__OTKApp";
    const char* m_closestHitName = "__closesthit__OTKApp";
    const char* m_anyhitName = nullptr;
    const char* m_intersectName = nullptr;

    unsigned int m_optixGeometryFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
};

void setGLFWCallbacks( OTKApp* app );

} // namespace otkApp
