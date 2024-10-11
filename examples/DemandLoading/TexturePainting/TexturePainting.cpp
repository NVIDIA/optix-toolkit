// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

//------------------------------------------------------------------------------
// TexturePainting 
// demonstrates the following demand texture features:
// * Preinitializing texture samplers
// * Reloading individual texture tiles
// * Invalidating all of the resident tiles in a texture
// * Swapping images in a live texture
//------------------------------------------------------------------------------

#include "CanvasImage.h"
#include "TexturePaintingKernelCuda.h"
#include "TexturePaintingParams.h"

#include <OptiXToolkit/OTKAppBase/OTKApp.h>
#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>

using namespace otkApp;
using namespace demandLoading;

class TexturePaintingApp : public OTKApp
{
  public:
    TexturePaintingApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
        : OTKApp( appTitle, width, height, outFileName, glInterop )
    {
        m_brush.set( m_brushWidth, m_brushWidth, brushColor( m_brushColorVal, m_brushAlpha ) );
    }
    void createScene();
    void createTexture();
    void initTexture();

    // GLFW callbacks
    void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods ) override;
    virtual void cursorPosCallback( GLFWwindow* window, double xpos, double ypos ) override;
    virtual void pollKeys() override;
    virtual void keyCallback( GLFWwindow* window, int key, int scancode, int action, int mods ) override;

  protected:
    // Canvas
    const static unsigned int NUM_CANVASES = 4;
    float4 m_canvasBackgroundColor = float4{ 1.0f, 1.0f, 1.0f, 0.0f };
    std::shared_ptr<imageSource::CanvasImage> m_canvases[NUM_CANVASES];
    unsigned int m_activeCanvas = 0;
    
    // Brush
    const static int MIN_BRUSH_SIZE = 10;
    const static int MAX_BRUSH_SIZE = 100;
    const static int BRUSH_SIZE_INC = 10;
    int m_brushWidth = 30;
    float m_brushColorVal = 0.0f;
    float m_brushAlpha = 0.2f;
    imageSource::CanvasBrush m_brush;

    // Texture 
    const demandLoading::DemandTexture* m_texture;

    void initLaunchParams( OTKAppPerDeviceOptixState& state, unsigned int numDevices ) override;
    int2 mouseToImageCoords( int mx, int my );
    float4 brushColor( float color, float a );
    void reloadDirtyTiles();
    void clearImage();
    void replaceTexture( unsigned int newCanvasId );
};

// std::min/max requires references to these static members.
const int TexturePaintingApp::MIN_BRUSH_SIZE;
const int TexturePaintingApp::MAX_BRUSH_SIZE;

void TexturePaintingApp::createScene()
{
    OTKAppTriangleHitGroupData mat{};
    std::vector<Vert> shape;

    // Square
    mat.tex = makeSurfaceTex( 0xffffff, 0, 0x000000, -1, 0x000000, -1, 0.1f, 0.0f );
    m_materials.push_back( mat );
    OTKAppShapeMaker::makeAxisPlane( float3{0, 0, 0}, float3{1, 1, 0}, shape );
    addShapeToScene( shape, m_materials.size() - 1 );

    copyGeometryToDevice();
}

void TexturePaintingApp::createTexture()
{
    // Create the backing images
    int canvasSize = m_windowHeight;
    for( unsigned int i=0; i<NUM_CANVASES; ++i)
    {
        m_canvases[i].reset( new imageSource::CanvasImage( canvasSize, canvasSize ) );
        m_canvases[i]->clearImage( m_canvasBackgroundColor );
    }

    demandLoading::TextureDescriptor texDesc = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, FILTER_BILINEAR );

    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        // Create a single texture that will switch between canvases
        m_texture = &state.demandLoader->createTexture( m_canvases[m_activeCanvas], texDesc );
        m_textureIds.push_back( m_texture->getId() );
    }
}

void TexturePaintingApp::initTexture()
{
    // Initialize the texture samplers for each device (not required, but saves a launch)
    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        CUcontext context;
        OTK_ERROR_CHECK( cuStreamGetCtx( state.stream, &context ) );
        OTK_ERROR_CHECK( cuCtxSetCurrent( context ) );

        state.demandLoader->initTexture( state.stream, m_texture->getId() );
    }
}

void TexturePaintingApp::mouseButtonCallback( GLFWwindow* window, int button, int action, int /*mods*/ )
{
    glfwGetCursorPos( window, &m_mousePrevX, &m_mousePrevY );
    m_mouseButton = ( action == GLFW_PRESS ) ? button : NO_BUTTON;

    // Map cursor position to button index
    unsigned int buttonIndex = 0xFFFFFFFF;
    if( m_mousePrevY >= m_windowHeight - (BUTTON_SIZE + BUTTON_SPACING) )
    {
        buttonIndex = static_cast<unsigned int>( m_mousePrevX / ( BUTTON_SIZE + BUTTON_SPACING ) );
    }

    if( m_mouseButton == GLFW_MOUSE_BUTTON_LEFT && buttonIndex < NUM_CANVASES )
    {
        replaceTexture( buttonIndex );
    }
    else if( m_mouseButton == GLFW_MOUSE_BUTTON_LEFT )
    {
        int2 p = mouseToImageCoords( static_cast<int>( m_mousePrevX ), static_cast<int>( m_mousePrevY ) );
        m_canvases[m_activeCanvas]->drawBrush( m_brush, p.x, p.y );
        reloadDirtyTiles();
    }
}

void TexturePaintingApp::cursorPosCallback( GLFWwindow* /*window*/, double xpos, double ypos )
{
    const int BRUSH_SIZE_CHANGE_SPEED = 2;
    const float BRUSH_COLOR_CHANGE_SPEED = 20.0f;

    if( m_mouseButton == GLFW_MOUSE_BUTTON_LEFT ) 
    {
        int2 p0 = mouseToImageCoords( static_cast<int>( m_mousePrevX ), static_cast<int>( m_mousePrevY ) );
        int2 p1 = mouseToImageCoords( static_cast<int>( xpos ), static_cast<int>( ypos ) );
        m_canvases[m_activeCanvas]->drawStroke( m_brush, p0.x, p0.y, p1.x, p1.y );
        reloadDirtyTiles();
    }
    else if( m_mouseButton == GLFW_MOUSE_BUTTON_RIGHT )
    {
        // brush size
        m_brushWidth -= int( ypos - m_mousePrevY ) / BRUSH_SIZE_CHANGE_SPEED;
        m_brushWidth = std::min( m_brushWidth, MAX_BRUSH_SIZE );
        m_brushWidth = std::max( m_brushWidth, MIN_BRUSH_SIZE );
        // brush color
        m_brushColorVal += static_cast<float>( xpos - m_mousePrevX ) / BRUSH_COLOR_CHANGE_SPEED;
        m_brush.set( m_brushWidth, m_brushWidth, brushColor( m_brushColorVal, m_brushAlpha ) );
    }

    m_mousePrevX = xpos;
    m_mousePrevY = ypos;
}

void TexturePaintingApp::pollKeys()
{
}

void TexturePaintingApp::keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods )
{
    OTKApp::keyCallback( window, key, scancode, action, mods );
    if( action != GLFW_PRESS )
        return;

    if( key == GLFW_KEY_X )
    {
        clearImage();
    }
    else if( key >= GLFW_KEY_1 && key < (int32_t)( GLFW_KEY_1 + NUM_CANVASES ) )
    {
        replaceTexture( key - GLFW_KEY_1 );
    }
    else if( key == GLFW_KEY_UP )
    {
        m_brushWidth = std::min( MAX_BRUSH_SIZE, m_brush.m_width + BRUSH_SIZE_INC );
        m_brush.set( m_brushWidth, m_brushWidth, brushColor( m_brushColorVal, m_brushAlpha ) );
    }
    else if( key == GLFW_KEY_DOWN )
    {
        m_brushWidth = std::max( MIN_BRUSH_SIZE, m_brush.m_width - BRUSH_SIZE_INC );
        m_brush.set( m_brushWidth, m_brushWidth, brushColor( m_brushColorVal, m_brushAlpha ) );
    }
    else if( key == GLFW_KEY_LEFT )
    {
        m_brushColorVal--;
        m_brush.set( m_brushWidth, m_brushWidth, brushColor( m_brushColorVal, m_brushAlpha ) );
    }
    else if( key == GLFW_KEY_RIGHT )
    {
        m_brushColorVal++;
        m_brush.set( m_brushWidth, m_brushWidth, brushColor( m_brushColorVal, m_brushAlpha ) );
    }
}

void TexturePaintingApp::initLaunchParams( OTKAppPerDeviceOptixState& state, unsigned int numDevices )
{
    OTKApp::initLaunchParams( state, numDevices );

    // Use extra data for params specific to texture painting.
    TexturePaintingParams* tp = reinterpret_cast<TexturePaintingParams*>( state.params.extraData );
    tp->numCanvases = NUM_CANVASES;
    tp->activeCanvas = m_activeCanvas;
    tp->brushWidth = m_brush.m_width;
    tp->brushHeight = m_brush.m_height;
    tp->brushColor = m_brush.m_color;
    tp->brushColor.w = 1.0f;
}

int2 TexturePaintingApp::mouseToImageCoords( int mx, int my )
{
    int imageWidth = m_canvases[m_activeCanvas]->getInfo().width;
    int imageHeight = m_canvases[m_activeCanvas]->getInfo().height;

    float x = 0.5f + ( mx - 0.5f * m_windowWidth ) / m_windowHeight;
    float y = 1.0f - static_cast<float>( my ) / static_cast<float>( m_windowHeight );
    return int2{ static_cast<int>( x * imageWidth ), static_cast<int>( y * imageHeight ) };
}

float4 TexturePaintingApp::brushColor( float color, float a )
{
    const float s = 0.85f;
    int c = int( floorf( color ) );
    // Construct an RGB color from the 3 low order bits of c.
    return float4{s * float( c & 1 ), s * float( ( c >> 1 ) & 1 ), s * float( ( c >> 2 ) & 1 ), a};
}

void TexturePaintingApp::reloadDirtyTiles()
{
    std::set<int>& dirtyTiles = m_canvases[m_activeCanvas]->getDirtyTiles();

    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        cudaSetDevice( state.device_idx );
        // For each dirty tile that is resident, have the demand loader reload it. 
        for (std::set<int>::iterator it = dirtyTiles.begin(); it != dirtyTiles.end(); ++it)
        {
            int3 tileCoord = m_canvases[m_activeCanvas]->unpackTileId( *it );
            unsigned int pageId = state.demandLoader->getTextureTilePageId( m_texture->getId(), tileCoord.z, tileCoord.x, tileCoord.y );
            if( state.demandLoader->pageResident( pageId ) )
                 state.demandLoader->loadTextureTile( state.stream,  m_texture->getId(), tileCoord.z, tileCoord.x, tileCoord.y );
        }
    }

    m_canvases[m_activeCanvas]->clearDirtyTiles();
}

void TexturePaintingApp::clearImage()
{
    // Clear the current canvas image, and unload all of the resident tiles in the texture.
    m_canvases[m_activeCanvas]->clearImage( m_canvasBackgroundColor ); 
    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        cudaSetDevice( state.device_idx );
        state.demandLoader->unloadTextureTiles( m_textureIds[0] );
    }
}

void TexturePaintingApp::replaceTexture( unsigned int newCanvasId )
{
    if( newCanvasId != m_activeCanvas )
    {
        // Switch the image that the texture is using, and unload all of the resident texture tiles.
        m_activeCanvas = newCanvasId;
        for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
        {
            cudaSetDevice( state.device_idx );
            demandLoading::TextureDescriptor texDesc = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, FILTER_BILINEAR );
            state.demandLoader->replaceTexture( state.stream, m_textureIds[0], m_canvases[m_activeCanvas], texDesc, false );
        }
    }
}

//------------------------------------------------------------------------------
// Usage
//------------------------------------------------------------------------------

void printUsage( const char* program )
{
    // clang-format off
    std::cerr << "\nUsage: " << program << " [options]\n"
        "\n"
        "Options:\n"
        "   --dim=<width>x<height>   Window dimensions.\n"
        "   --no-gl-interop          Disable OpenGL interop.\n";
    // clang-format on

    exit(0);
}

void printKeyCommands()
{
    // clang-format off
    std::cout <<
        "Keyboard:\n"
        "   <ESC>:         exit\n"
        "   [1,2,3,4]:     set canvas\n"
        "   [x]:           clear current canvas\n"
        "   [arrow keys]:  brush color/size\n"
        "\n"
        "Mouse:\n"
        "   <LMB>:  draw\n"
        "   <RMB>:  brush color/size\n"
        "\n";
    // clang-format on
}

//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    int         windowWidth  = 950;
    int         windowHeight = 700;
    const char* outFileName  = "";
    bool        glInterop    = true;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        const bool        lastArg = ( i == argc - 1 );

        if( ( arg == "--file" ) && !lastArg )
            outFileName = argv[++i];
        else if( arg.substr( 0, 6 ) == "--dim=" )
            otk::parseDimensions( arg.substr( 6 ).c_str(), windowWidth, windowHeight );
        else if( arg == "--no-gl-interop" )
            glInterop = false;
        else
            printUsage( argv[0] );
    }

    printKeyCommands();

    TexturePaintingApp app( "Texture Painting", windowWidth, windowHeight, outFileName, glInterop );
    app.initDemandLoading( demandLoading::Options{} );
    app.createTexture();
    app.createScene();
    app.initOptixPipelines( TexturePaintingCudaText(), TexturePaintingCudaSize );
    app.initTexture();
    app.startLaunchLoop();
    app.printDemandLoadingStats();
    
    return 0;
}
