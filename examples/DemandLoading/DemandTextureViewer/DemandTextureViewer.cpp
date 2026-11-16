// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <memory>
#include <stdexcept>

#include <DemandTextureViewerKernelCuda.h>

#include <OptiXToolkit/DemandLoading/DemandLoadLogger.h>
#include <OptiXToolkit/OTKAppBase/OTKApp.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/ImageSource/MipMapImageSource.h>
#include <OptiXToolkit/ImageSource/TiledImageSource.h>

using namespace otkApp;

//------------------------------------------------------------------------------
// DemandTextureViewer
// Shows basic use of OptiX demand textures.
//------------------------------------------------------------------------------

using ImageSourcePtr = std::shared_ptr<imageSource::ImageSource>;

class DemandTextureViewer : public OTKApp
{
  public:
    DemandTextureViewer( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
        : OTKApp( appTitle, width, height, outFileName, glInterop, UI_IMAGEVIEW )
    {
    }
    void createTexture( const std::string& textureName, bool tile, bool mipmap );
    void createScene();

  private:
    ImageSourcePtr createImageSource( const std::string& textureName, bool tile, bool mipmap );
};

inline bool endsWith( const std::string& text, const std::string& suffix )
{
    return text.length() >= suffix.length() && text.substr( text.length() - suffix.length() ) == suffix;
}

ImageSourcePtr DemandTextureViewer::createImageSource( const std::string& textureName, bool tile, bool mipmap )
{
    ImageSourcePtr img = imageSource::createImageSource( textureName );
    if( !img )
    {
        throw std::runtime_error( "Could not create requested texture " +
            ( textureName.empty() ? std::string{} : " (" + textureName + ")" ) );
    }

    imageSource::TextureInfo texInfo;
    img->open( &texInfo );
    if( mipmap && texInfo.numMipLevels <= 1 )
    {
        img = createMipMapImageSource( img );
    }
    if( tile  && !texInfo.isTiled )
    {
        img = createTiledImageSource( img );
    }
    return img;
}

void DemandTextureViewer::createTexture( const std::string& textureName, bool tile, bool mipmap )
{
    ImageSourcePtr imageSource( createImageSource( textureName, tile, mipmap ) );

    demandLoading::TextureDescriptor texDesc = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, FILTER_BILINEAR );
    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        const demandLoading::DemandTexture& texture = state.demandLoader->createTexture( imageSource, texDesc );
        if( m_textureIds.empty() )
            m_textureIds.push_back( texture.getId() );
    }
}

void DemandTextureViewer::createScene()
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

//------------------------------------------------------------------------------
// Usage
//------------------------------------------------------------------------------

void printUsage( const char* program )
{
    // clang-format off
    std::cerr << "\nUsage: " << program << " [options]\n"
        "\n"
        "Options:\n"
        "   --texture <texturefile|checkerboard|mandelbrot>     Use texture image file.\n"
        "   --dim=<width>x<height>      Specify rendering dimensions.\n"
        "   --file <outputfile>         Render to output file and exit.\n"
        "   --tile                      Make image tileable\n"
        "   --mipmap                    Make image mipmapped\n"
        "   --log <level>               Set demand load log level (0-10)\n"
        "   --no-gl-interop             Disable OpenGL interop.\n";
    // clang-format on

    exit(0);
}

void printKeyCommands()
{
    // clang-format off
    std::cout <<
        "Keyboard:\n"
        "   <ESC>:  exit\n"
        "   WASD:   pan\n"
        "   QE:     zoom\n"
        "   C:      recenter\n"
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
    int         logLevel = 0;

    std::string textureName = "checkerboard";
    bool        tile = false;
    bool        mipmap = false;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        const bool        lastArg = ( i == argc - 1 );

        if( arg == "--texture" && !lastArg )
            textureName = argv[++i];
        else if( arg == "--file" && !lastArg )
            outFileName = argv[++i];
        else if( arg.substr( 0, 6 ) == "--dim=" )
            otk::parseDimensions( arg.substr( 6 ).c_str(), windowWidth, windowHeight );
        else if( arg == "--no-gl-interop" )
            glInterop = false;
        else if( arg == "--tile" )
            tile = true;
        else if( arg == "--mipmap" )
            mipmap = true;
        else if( arg == "--log" && !lastArg )
            logLevel = atoi( argv[++i] );
        else
            printUsage( argv[0] );
    }

    printKeyCommands();
    DemandLoadLogger::setLogFunction( standardDemandLoadLogCallback, logLevel );

    DemandTextureViewer app( "Demand Texture Viewer", windowWidth, windowHeight, outFileName, glInterop );
    app.initDemandLoading( demandLoading::Options{} );
    app.createTexture( textureName, tile, mipmap );
    app.createScene();
    app.initOptixPipelines( DemandTextureViewerCudaText(), DemandTextureViewerCudaSize );
    app.startLaunchLoop();
    app.printDemandLoadingStats();

    return 0;
}