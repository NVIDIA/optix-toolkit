// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

// Suppress deprecation warnings from CUDA vector types in ShaderUtil headers
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <memory>
#include <stdexcept>

#include <NeuralTextureViewerKernelCuda.h>

#include <OptiXToolkit/OTKAppBase/OTKApp.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/DemandLoading/TextureSampler.h>
#include <OptiXToolkit/NeuralTextures/NeuralTextureSource.h>

#include "SourceDir.h"  // generated from SourceDir.h.in

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

using namespace neuralTextures;
using namespace otkApp;

//------------------------------------------------------------------------------
// NeuralTextureViewer
// Display neural textures
//------------------------------------------------------------------------------

using ImageSourcePtr = std::shared_ptr<imageSource::ImageSource>;

class NeuralTextureViewer : public OTKApp
{
  public:
    NeuralTextureViewer( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
        : OTKApp( appTitle, width, height, outFileName, glInterop, UI_IMAGEVIEW )
    {
        m_render_mode = 1;
    }
    void createTextures( const std::vector<std::string>& textureNames, int filterMode );
    void createScene();

  private:
    void keyCallback( GLFWwindow* window, int key, int scancode, int action, int mods ) override;
};

inline bool endsWith( const std::string& text, const std::string& suffix )
{
    return text.length() >= suffix.length() && text.substr( text.length() - suffix.length() ) == suffix;
}

void NeuralTextureViewer::createTextures( const std::vector<std::string>& textureNames, int filterMode )
{
    std::vector< demandLoading::TextureDescriptor > subTexDescs;
    std::vector< std::shared_ptr<imageSource::ImageSource> > subImageSources;

    for( unsigned int i = 0; i < textureNames.size(); ++i )
    {
        demandLoading::TextureDescriptor texDesc = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, (FilterMode)filterMode );
        texDesc.flags |= CU_TRSF_READ_AS_INTEGER;
        subTexDescs.push_back( texDesc );

        NeuralTextureSource* neuralTextureSource = new NeuralTextureSource( textureNames[i] );
        subImageSources.push_back( ImageSourcePtr( neuralTextureSource ) );
    }

    int baseTextureId = -1; // no base texture
    int udim = ( textureNames.size() > 1 ) ? 10 : 1;
    int vdim = ( textureNames.size() > 1 ) ? 10 : 1;

    // Create a udim texture for all the devices
    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        createContext( state );
        state.params.extraData[0] = udim;
        state.params.extraData[1] = vdim;

        const demandLoading::DemandTexture& udimTexture =
            state.demandLoader->createUdimTexture( subImageSources, subTexDescs, udim, vdim, baseTextureId );

        if( m_textureIds.empty() )
            m_textureIds.push_back( udimTexture.getId() );
    } 
}

void NeuralTextureViewer::createScene()
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

void NeuralTextureViewer::keyCallback( GLFWwindow* window, int key, int scancode, int action, int mods )
{
    OTKApp::keyCallback( window, key, scancode, action, mods );
    m_subframeId = 0;
}

//------------------------------------------------------------------------------
// Usage
//------------------------------------------------------------------------------

void printUsage( const char* program )
{
    // clang-format off
    std::cerr << "\nUsage: " << program << " [options] [ntcTextureFiles]\n"
        "\n"
        "Options:\n"
        "   --dim=<width>x<height>      Specify rendering dimensions.\n"
        "   --file <outputfile>         Render to output file and exit.\n"
        "   --point|--linear|--cubic    Stochastic filter mode.\n"
        "   --no-gl-interop             Disable OpenGL interop.\n";
    // clang-format on

    exit(0);
}

void printKeyCommands()
{
    // clang-format off
    std::cout <<
        "Keyboard:\n"
        "   <ESC>:   exit\n"
        "   WASD:    pan\n"
        "   QE:      zoom\n"
        "   C:       recenter\n"
        "   [1..4]:  texture from the set to show\n"
        "\n"
        "Mouse:\n"
        "   <LMB>:  pan\n"
        "   <RMB>:  zoom\n"
        "\n";
    // clang-format on
}

//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    int         windowWidth  = 768;
    int         windowHeight = 768;
    std::string outFileName = "";
    bool        glInterop = true;
    int         filterMode = FILTER_BILINEAR;

    std::vector<std::string> textureFiles;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        const bool        lastArg = ( i == argc - 1 );

        if( arg == "--file" && !lastArg )
            outFileName = argv[++i];
        else if( arg.substr( 0, 6 ) == "--dim=" )
            otk::parseDimensions( arg.substr( 6 ).c_str(), windowWidth, windowHeight );
        else if( arg == "--no-gl-interop" )
            glInterop = false;
        else if( endsWith( arg, ".ntc" ) )
            textureFiles.push_back( arg );
        else if( arg == "--point" )
            filterMode = FILTER_POINT;
        else if( arg == "--linear" )
            filterMode = FILTER_BILINEAR;
        else if( arg == "--cubic" )
            filterMode = FILTER_SMARTBICUBIC;
        else
            printUsage( argv[0] );
    }

    if( textureFiles.size() == 0 )
    {
        std::string defaultTexture = getSourceDir() + "/Textures/colors.ntc";
        textureFiles.push_back( defaultTexture );
    }

    printKeyCommands();

    demandLoading::Options dlOptions{};

    NeuralTextureViewer app( "Neural Texture Viewer", windowWidth, windowHeight, outFileName, glInterop );
    app.resetAccumulator();
    app.setNumLaunches( 32 );
    app.initDemandLoading( dlOptions );
    app.createTextures( textureFiles, filterMode );
    app.createScene();
    app.initOptixPipelines( NeuralTextureViewerCudaText(), NeuralTextureViewerCudaSize );
    app.startLaunchLoop();
    app.printDemandLoadingStats();

    return 0;
}