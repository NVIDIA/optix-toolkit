// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda_runtime.h>

#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Gui/GLDisplay.h>
#include <OptiXToolkit/Gui/glfw3.h>
#include <OptiXToolkit/Util/EXRInputFile.h>

#include "PerDeviceOptixState.h"

namespace ommBakingApp
{

struct CudaTexture
{
public:
    CudaTexture() {};
    CudaTexture( const CudaTexture& ) /*= delete; */ { assert( false ); }
    CudaTexture( CudaTexture&& source ) noexcept
    { swap( source ); }
    ~CudaTexture() { destroy(); }

    CudaTexture( std::string textureName );

    CudaTexture( otk::EXRInputFile* imageFile )
    { create( imageFile ); }

    void destroy()
    {
        if( m_texObj )
        {
            cudaDestroyTextureObject( m_texObj );
            m_texObj = {};
        }
        m_data.free();
    }

    void create( otk::EXRInputFile* imageFile );

    CudaTexture& operator=( const CudaTexture& source ) = delete;

    CudaTexture& operator=( CudaTexture&& source ) noexcept
    {
        swap( source );
        return *this;
    }

    void swap( CudaTexture& source )
    {
        std::swap( m_texObj, source.m_texObj );
        std::swap( m_data, source.m_data );
    }

    cudaTextureObject_t get() const
    {
        return m_texObj;
    }

 private:
    cudaTextureObject_t   m_texObj = {};
    CuPitchedBuffer<char> m_data = {};
};

class OmmBakingApp
{
  public:
    OmmBakingApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop );
    virtual ~OmmBakingApp();

    // Public functions to initialize the app and start rendering
    void startLaunchLoop();

    // GLFW callbacks
    virtual void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods );
    virtual void cursorPosCallback( GLFWwindow* window, double xpos, double ypos );
    virtual void windowSizeCallback( GLFWwindow* window, int width, int height );
    virtual void pollKeys();
    virtual void keyCallback( GLFWwindow* window, int key, int scancode, int action, int mods );
    GLFWwindow* getWindow() { return m_window; }

  protected:

    void initOptixPipelines( const char* moduleCode, const size_t moduleCodeSize, int numDevices );

    // OptiX setup
    virtual void createContext( PerDeviceOptixState& state );
    virtual void createModule( PerDeviceOptixState& state, const char* moduleCode, size_t codeSize );
    virtual void createProgramGroups( PerDeviceOptixState& state );
    virtual void createPipeline( PerDeviceOptixState& state );
    virtual void cleanupState( PerDeviceOptixState& state );

    // Hooks
    virtual void buildAccel( const PerDeviceOptixState& state ) = 0;
    virtual void createSBT( const PerDeviceOptixState& state ) = 0;
    virtual void performLaunch( const PerDeviceOptixState& state, uchar4* result_buffer ) = 0;

    // OptiX launches
    void initView();
    void performLaunches();

    // Displaying and saving images
    void displayFrame();
    void saveImage();
    bool isInteractive() const { return m_outputFileName.empty(); }

  protected:

    int32_t getWidth() const { return m_outputBuffer->width(); }
    int32_t getHeight() const { return m_outputBuffer->height(); }
    float3 getEye() const { return m_eye; }
    float2 getViewDims() const { return m_viewDims; }

  private:

    // Viewpoint description
    const float3 INITIAL_LOOK_FROM{ 0.5f, 0.5f, 1.0f };
    const float INITIAL_VIEW_DIM = 1.0f;

    float2 m_viewDims = float2{ INITIAL_VIEW_DIM, INITIAL_VIEW_DIM };
    float3 m_eye = INITIAL_LOOK_FROM;

    // Window and output buffer
    GLFWwindow* m_window = nullptr;
    std::unique_ptr<otk::CUDAOutputBuffer<uchar4>> m_outputBuffer;
    std::unique_ptr<otk::GLDisplay>                m_glDisplay;
    int                                            m_windowWidth;
    int                                            m_windowHeight;
    std::string                                    m_outputFileName = "";

    // OptiX states for each device
    std::vector<PerDeviceOptixState> m_perDeviceOptixStates;

    // Mouse state
    static const int NO_BUTTON = -1;

    double m_mousePrevX  = 0;
    double m_mousePrevY  = 0;
    int    m_mouseButton = NO_BUTTON;

    bool m_glInterop = false;
};

void setGLFWCallbacks( OmmBakingApp* app );

} // namespace ommBakingApp
