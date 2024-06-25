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

// This include is needed to avoid a link error
#include <optix_stubs.h>

#include "StochasticTextureFilteringKernel.h"

#include <OptiXToolkit/DemandTextureAppBase/DemandTextureApp3D.h>
#include <OptiXToolkit/DemandTextureAppBase/ShapeMaker.h>
#include <OptiXToolkit/ImageSources/MultiCheckerImage.h>

#include <OptiXToolkit/ShaderUtil/ray_cone.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/Gui/Gui.h>
#include <OptiXToolkit/Gui/glfw3.h>

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "StochasticTextureFilteringParams.h"

using namespace otk;
using namespace demandTextureApp;
using namespace demandLoading;
using namespace imageSource;

//------------------------------------------------------------------------------
// StochasticTextureFilteringApp
//------------------------------------------------------------------------------

class StochasticTextureFilteringApp : public DemandTextureApp3D
{
  public:
    StochasticTextureFilteringApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop );
    void setTextureName( const char* textureName ) { m_textureName = textureName; }
    void createTexture() override;
    void initView() override;
    void createScene();
    void setSceneId( int sceneId ) { m_sceneId = sceneId; }
    void initLaunchParams( PerDeviceOptixState& state, unsigned int numDevices ) override;
    void drawGui() override;

  protected:
    std::string m_textureName;
    int m_sceneId = 0;

    int m_minRayDepth = 0;
    int m_maxRayDepth = 6;
    int m_updateRayCones = 1;

    // GUI controls
    unsigned int m_selectedOutputValueId = 0;
    unsigned int m_selectedPixelFilterId = 0;
    unsigned int m_selectedTextureFilterId = 0;
    unsigned int m_selectedTextureJitterId = 0;
    bool         m_singleSample = false;
    float        m_filterWidth = 1.0f;
    float        m_filterStrength = 1.0f;

    void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods ) override;
    void keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods ) override;

};


StochasticTextureFilteringApp::StochasticTextureFilteringApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
    : DemandTextureApp3D( appTitle, width, height, outFileName, glInterop )
{
    m_backgroundColor = float4{1.0f, 1.0f, 1.0f, 0.0f};
    m_projection = Projection::PINHOLE;
    m_render_mode = 1;
}


void StochasticTextureFilteringApp::initView()
{
    if( m_sceneId == 0 )
        setView( float3{0.0f, 0.0f, 1.0f}, float3{-10.0f, -5.0f, 0.0f}, float3{0.0f, 0.0f, 1.0f}, 30.0f );
    else if( m_sceneId == 1 )
        setView( float3{0.0f, 40.0f, 5.0f}, float3{0.0f, 0.0f, 0.0f}, float3{0.0f, 0.0f, 1.0f}, 30.0f );
    else 
        setView( float3{0.0f, 25.0f, 7.0f}, float3{0.0f, 0.0f, 3.0f}, float3{0.0f, 0.0f, 1.0f}, 30.0f );
}


void StochasticTextureFilteringApp::initLaunchParams( PerDeviceOptixState& state, unsigned int numDevices )
{
    // If the GUI state has changed, reset the subframe id.
    if( state.params.i[PIXEL_FILTER_ID] != static_cast<int>( m_selectedPixelFilterId ) ||
        state.params.i[TEXTURE_FILTER_ID] != static_cast<int>( m_selectedTextureFilterId ) ||
        state.params.i[TEXTURE_JITTER_ID] != static_cast<int>( m_selectedTextureJitterId ) ||
        state.params.f[TEXTURE_FILTER_WIDTH_ID] != m_filterWidth ||
        state.params.f[TEXTURE_FILTER_STRENGTH_ID] != m_filterStrength ||
        m_singleSample )
        m_subframeId = 0;

    DemandTextureApp::initLaunchParams( state, numDevices );

    state.params.i[SUBFRAME_ID]       = m_subframeId;
    state.params.i[PIXEL_FILTER_ID]   = m_selectedPixelFilterId; 
    state.params.i[TEXTURE_FILTER_ID] = m_selectedTextureFilterId;
    state.params.i[TEXTURE_JITTER_ID] = m_selectedTextureJitterId;
    state.params.i[MOUSEX_ID]         = static_cast<int>( m_mousePrevX );
    state.params.i[MOUSEY_ID]         = static_cast<int>( m_mousePrevY );

    state.params.f[MIP_SCALE_ID]               = m_mipScale;
    state.params.f[TEXTURE_FILTER_WIDTH_ID]    = m_filterWidth;
    state.params.f[TEXTURE_FILTER_STRENGTH_ID] = m_filterStrength;
}

void StochasticTextureFilteringApp::createTexture()
{
    std::shared_ptr<ImageSource> imageSource( createExrImage( m_textureName.c_str() ) );
    if( !imageSource && !m_textureName.empty() )
        std::cout << "ERROR: Could not find image " << m_textureName << ". Substituting procedural image.\n";
    if( !imageSource )
        imageSource.reset( new imageSources::MultiCheckerImage<uchar4>( 16384, 16384, 256, true, false ) );
    
    demandLoading::TextureDescriptor texDesc0 = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, FILTER_POINT );
    demandLoading::TextureDescriptor texDesc1 = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, FILTER_BILINEAR );

    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        const demandLoading::DemandTexture& texture = state.demandLoader->createTexture( imageSource, texDesc0 );
        if( m_textureIds.empty() )
            m_textureIds.push_back( texture.getId() );
        state.demandLoader->createTexture( imageSource, texDesc1 );
    }
}

void StochasticTextureFilteringApp::createScene()
{
    const unsigned int NUM_SEGMENTS = 128;
    TriangleHitGroupData mat{};
    std::vector<Vert> shape;

    // ground plane
    if( m_sceneId == 0 )
    {
        mat.tex = makeSurfaceTex( 0x999999, 0, 0x000000, -1, 0x000000, -1, 0.1f, 0.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeAxisPlane( float3{-100, -100, 0}, float3{100, 100, 0}, shape );
        addShapeToScene( shape, m_materials.size() - 1 );
    }

    // square
    else if( m_sceneId == 1 )
    {
        mat.tex = makeSurfaceTex( 0x999999, 0, 0x000000, -1, 0x000000, -1, 0.1f, 0.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeAxisPlane( float3{-10, 0, -10}, float3{10, 0, 10}, shape );
        addShapeToScene( shape, m_materials.size() - 1 );
    }

    // vase
    else if( m_sceneId >= 2 )
    {
        // Ground
        mat.tex = makeSurfaceTex( 0x222222, -1, 0x111111, -1, 0x000000, -1, 0.01, 0.0f );
        m_materials.push_back( mat );

        ShapeMaker::makeAxisPlane( float3{-40, -40, 0}, float3{40, 40, 0}, shape );
        addShapeToScene( shape, m_materials.size() - 1 );

        // Vase
        mat.tex = makeSurfaceTex( 0x999999, 0, 0x111111, -1, 0x000000, -1, 0.0001, 0.0f );
        m_materials.push_back( mat );

        ShapeMaker::makeVase( float3{0.0f, 0.0f, 0.01f}, 1.0f, 4.0f, 8.0f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() -1 );

        ShapeMaker::makeSphere( float3{-5.0f, 1.0f, 0.7f}, 0.7f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() - 1 );

        // Vase liners with diffuse material to block negative curvature traps
        mat.tex = makeSurfaceTex( 0x111111, -1, 0x111111, -1, 0x000000, -1, 0.1, 0.0f );
        m_materials.push_back( mat );

        ShapeMaker::makeVase( float3{0.0f, 0.0f, 0.01f}, 0.99f, 3.99f, 8.0f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() -1 );
    }

    copyGeometryToDevice();
}

void StochasticTextureFilteringApp::mouseButtonCallback( GLFWwindow* window, int button, int action, int /*mods*/ )
{
    ImGuiIO& io = ImGui::GetIO();
    io.AddMouseButtonEvent( button, (bool) action );

    if( !io.WantCaptureMouse )
    {
        glfwGetCursorPos( window, &m_mousePrevX, &m_mousePrevY );
        m_mouseButton = ( action == GLFW_PRESS ) ? button : NO_BUTTON;
    }
}

void StochasticTextureFilteringApp::keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods )
{
    DemandTextureApp::keyCallback( window, key, scancode, action, mods );
    if( action != GLFW_PRESS )
        return;

    if( key == GLFW_KEY_LEFT ) {
        m_maxRayDepth = std::max( m_maxRayDepth - 1, 1 );
    } else if( key == GLFW_KEY_RIGHT ) {
        m_maxRayDepth++;
    } else if( key == GLFW_KEY_UP ) {
        m_minRayDepth = std::min( m_minRayDepth + 1, m_maxRayDepth );
    } else if( key == GLFW_KEY_DOWN ) {
        m_minRayDepth = std::max( m_minRayDepth - 1, 0 );
    } else if( key == GLFW_KEY_EQUAL ) {
        m_mipScale *= 0.5f;
    } else if( key == GLFW_KEY_MINUS ) {
        m_mipScale *= 2.0f;
    } else if( key == GLFW_KEY_X ) {
        for( PerDeviceOptixState& state : m_perDeviceOptixStates )
        {
            OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
            state.demandLoader->unloadTextureTiles( m_textureIds[0] );
        }
    } else if( key == GLFW_KEY_U ) {
        m_updateRayCones = static_cast<int>( !m_updateRayCones );
    } else if( key == GLFW_KEY_P && m_projection == Projection::PINHOLE ) {
        m_projection = Projection::THINLENS;
    } else if( key == GLFW_KEY_P && m_projection == Projection::THINLENS ) {
        m_projection = Projection::PINHOLE;
    } else if( key == GLFW_KEY_O ) {
        m_lens_width *= 1.1f;
    } else if ( key == GLFW_KEY_I ) {
        m_lens_width /= 1.1f;
    } else if( key == GLFW_KEY_F1 ) {
        saveImage();
    }

    m_subframeId = 0;
}

void displayComboBox( const char* title, const char* items[], unsigned int numItems, unsigned int& selectedId )
{
    if( ImGui::BeginCombo( title, items[selectedId], ImGuiComboFlags_HeightLarge ) )
    {
        for( unsigned int id = 0; id < numItems; ++id )
        {
            bool isSelected = ( selectedId == id ); 
            if( ImGui::Selectable( items[id], selectedId ) )
                selectedId = id;
            if( isSelected )
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
}

void StochasticTextureFilteringApp::drawGui()
{
    otk::beginFrameImGui();

    ImGui::SetNextWindowPos( ImVec2( 5, 5 ) );
    ImGui::SetNextWindowSize( ImVec2( 0, 0 ) );
    ImGui::SetNextWindowBgAlpha( 0.75f );
    ImGui::Begin( "" );

    ImGui::Text("framerate: %.1f fps", ImGui::GetIO().Framerate);
    displayComboBox( "Pixel Filter", PIXEL_FILTER_MODE_NAMES, pfSIZE, m_selectedPixelFilterId );
    displayComboBox( "Filter Mode", TEXTURE_FILTER_MODE_NAMES, fmSIZE, m_selectedTextureFilterId );
    displayComboBox( "Jitter Kernel", TEXTURE_JITTER_MODE_NAMES, jmSIZE, m_selectedTextureJitterId );
    ImGui::Checkbox( "Single Sample", &m_singleSample );

    ImGui::SliderFloat("Filter Width", &m_filterWidth, 0.0f, 3.0f, "%.3f", 0);
    ImGui::SliderFloat("Filter Strength", &m_filterStrength, 0.0f, 3.0f, "%.3f", 0);

    otk::endFrameImGui();
}


//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------

void printUsage( const char* argv0 )
{
    std::cerr << "\n\nUsage: " << argv0 << " [options]\n\n";
    std::cout << "Options:  --scene [0-5], --texture <texturefile.exr>, --launches <numLaunches>\n";
    std::cout << "          --dim=<width>x<height>, --file <outputfile.ppm>, --no-gl-interop\n\n";
    std::cout << "Mouse:    <LMB>:          pan camera\n";
    std::cout << "          <RMB>:          rotate camera\n\n";
    std::cout << "Keyboard: <ESC>:          exit\n";
    std::cout << "          1-7:            set output variable\n";
    std::cout << "          <LEFT>,<RIGHT>: change max depth\n";
    std::cout << "          <UP>,<DOWN>:    change min depth\n";
    std::cout << "          WASD,QE:        pan camera\n";
    std::cout << "          J,L:            rotate camera\n";
    std::cout << "          C:              reset view\n";
    std::cout << "          +,-:            change mip bias\n";
    std::cout << "          P:              toggle thin lens camera\n";
    std::cout << "          U:              toggle distance-based vs. ray cones\n";
    std::cout << "          I,O:            change lens width\n";
    std::cout << "          X:              unload all texture tiles\n\n";
}

int main( int argc, char* argv[] )
{
    int         windowWidth  = 900;
    int         windowHeight = 600;
    const char* textureName  = "";
    const char* outFileName  = "";
    bool        glInterop    = true;
    int         numLaunches  = 256;
    int         sceneId      = 1;

    printUsage( argv[0] );

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        const bool        lastArg = ( i == argc - 1 );

        if( ( arg == "--texture" ) && !lastArg )
            textureName = argv[++i];
        else if( ( arg == "--file" ) && !lastArg )
            outFileName = argv[++i];
        else if( arg.substr( 0, 6 ) == "--dim=" )
            otk::parseDimensions( arg.substr( 6 ).c_str(), windowWidth, windowHeight );
        else if( arg == "--no-gl-interop" )
            glInterop = false;
        else if( arg == "--launches" && !lastArg )
            numLaunches = atoi( argv[++i] );
        else if( arg == "--scene" && !lastArg )
            sceneId = atoi( argv[++i] );
        else 
            exit(0);
    }

    StochasticTextureFilteringApp app( "Stochastic Texture Filtering", windowWidth, windowHeight, outFileName, glInterop );
    app.setSceneId( sceneId );
    app.initView();
    app.setNumLaunches( numLaunches );
    app.sceneIsTriangles( true );
    app.initDemandLoading();
    app.setTextureName( textureName );
    app.createTexture();
    app.createScene();
    app.resetAccumulator();
    app.initOptixPipelines( StochasticTextureFilteringCudaText(), StochasticTextureFilteringCudaSize );
    app.startLaunchLoop();
    app.printDemandLoadingStats();
    
    return 0;
}
