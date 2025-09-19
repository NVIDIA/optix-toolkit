// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <string>

#include "UdimTextureViewerKernelCuda.h"

#include <OptiXToolkit/OTKAppBase/OTKApp.h>
#include <OptiXToolkit/DemandLoading/TextureSampler.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/ImageSource/DeviceMandelbrotImage.h>
#include <OptiXToolkit/ImageSource/DDSImageReader.h>

using namespace otkApp;
using namespace demandLoading;

//------------------------------------------------------------------------------
// UdimTextureApp
// Shows how to create and use udim textures.
//------------------------------------------------------------------------------

class UdimTextureApp : public OTKApp
{
  public:
    UdimTextureApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
        : OTKApp( appTitle, width, height, outFileName, glInterop, UI_IMAGEVIEW )
    {
    }
    void setUdimParams( const char* textureName, int texWidth, int texHeight, int udim, int vdim, bool useBaseImage );
    void createTexture();
    void createScene();
    void setUseSrgb( bool srgb ) { m_useSRGB = srgb; }

    void initLaunchParams( OTKAppPerDeviceOptixState& state, unsigned int numDevices ) override;

  private:
    std::string m_textureName;
    int         m_texWidth     = 8192;
    int         m_texHeight    = 8192;
    int         m_udim         = 10;
    int         m_vdim         = 10;
    bool        m_useBaseImage = false;
    bool        m_useSRGB      = false;
};

void UdimTextureApp::setUdimParams( const char* textureName, int texWidth, int texHeight, int udim, int vdim, bool useBaseImage )
{
    m_textureName  = textureName;
    m_texWidth     = texWidth;
    m_texHeight    = texHeight;
    m_udim         = udim;
    m_vdim         = vdim;
    m_useBaseImage = useBaseImage;
}

void UdimTextureApp::createTexture()
{
    // Split m_textureName into file name and extension
    size_t dotPos = m_textureName.rfind( '.' );
    std::string textureNameBase = (dotPos == std::string::npos) ? m_textureName : m_textureName.substr(0, dotPos);
    std::string textureNameExtension = (dotPos == std::string::npos) ? "" : m_textureName.substr(dotPos);

    // Make optional base texture
    int baseTextureId = -1;
    if( m_useBaseImage )
    {
        std::shared_ptr<imageSource::ImageSource> baseImageSource;
        if( m_textureName != "mandelbrot" && m_textureName != "checkerboard" )
            baseImageSource = imageSource::createImageSource( m_textureName, "" );
        if( !baseImageSource && m_textureName == "checkerboard" )
            baseImageSource.reset( new imageSource::MultiCheckerImage<float4>( m_texWidth, m_texHeight, 32, true ) );
        if( !baseImageSource )
            baseImageSource.reset( new imageSource::DeviceMandelbrotImage( m_texWidth, m_texHeight, -2.0, -2.0, 2.0, 2.0 ) );

        demandLoading::TextureDescriptor texDesc = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, FILTER_BILINEAR );
        if( m_useSRGB )
            texDesc.flags = texDesc.flags | CU_TRSF_SRGB;

        // Create a base texture for all devices
        for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
        {
            OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
            const demandLoading::DemandTexture& baseTexture = state.demandLoader->createTexture( baseImageSource, texDesc );
            baseTextureId                                   = baseTexture.getId();
            if( m_textureIds.empty() )
                m_textureIds.push_back( baseTextureId );
        }
    }
    if( m_udim == 0 ) // just a regular texture
        return;

    // Make subimages and define UDIM texture 
    std::vector< demandLoading::TextureDescriptor > subTexDescs;
    std::vector< std::shared_ptr<imageSource::ImageSource> > subImageSources;
    for( int v = 0; v < m_vdim; ++v )
    {
        for( int u = 0; u < m_udim; ++u )
        {
            std::shared_ptr<imageSource::ImageSource> subImage;
            if( m_textureName != "mandelbrot" && m_textureName != "checkerboard" ) // loading exr images
            {
                int         udimNum      = 1000 + ( v * 10 ) + ( u + 1 );
                std::string subImageName = textureNameBase + "_" + std::to_string( udimNum ) + ".exr";
                subImage                 = imageSource::createImageSource( subImageName, "" );
            }
            if( !subImage && m_texWidth == 0 ) // mixing different image sizes
            {
                int maxAspect = 64;
                int w         = std::max( 4 << u, ( 4 << v ) / maxAspect );
                int h         = std::max( 4 << v, ( 4 << u ) / maxAspect );
                subImage.reset( new imageSource::MultiCheckerImage<float4>( w, h, 4, true ) );
            }
            if( !subImage && m_textureName == "checkerboard" )
            {
                subImage.reset( new imageSource::MultiCheckerImage<float4>( m_texWidth, m_texHeight, 32, true ) );
            }
            if( !subImage ) // many images of the same size
            {
                double xmin = -2.0 + 4.0 * u / m_udim;
                double xmax = -2.0 + 4.0 * ( u + 1.0 ) / m_udim;
                double ymin = -2.0 + 4.0 * v / m_vdim;
                double ymax = -2.0 + 4.0 * ( v + 1.0 ) / m_vdim;
                subImage.reset( new imageSource::DeviceMandelbrotImage( m_texWidth, m_texHeight, xmin, ymin, xmax, ymax ) );
            }
            subImageSources.emplace_back( subImage );

            // Note: Use address mode CU_TR_ADDRESS_MODE_BORDER for subimages in tex2DGradUdimBlend calls in OptiX programs.
            // (CU_TR_ADDRESS_MODE_CLAMP for tex2DGradUdim calls).
            subTexDescs.push_back( makeTextureDescriptor( CU_TR_ADDRESS_MODE_BORDER, FILTER_SMARTBICUBIC ) );
            if( m_useSRGB )
                subTexDescs.back().flags |= CU_TRSF_SRGB;
        }
    }

    // Create a udim texture for all the devices
    for( OTKAppPerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        const demandLoading::DemandTexture& udimTexture =
            state.demandLoader->createUdimTexture( subImageSources, subTexDescs, m_udim, m_vdim, baseTextureId );

        if( m_textureIds.empty() )
            m_textureIds.push_back( udimTexture.getId() );
    }
}

void UdimTextureApp::createScene()
{
    OTKAppTriangleHitGroupData mat{};
    std::vector<Vert> shape;

    float udim = (m_udim >= 1) ? m_udim : 1.0f;
    float vdim = (m_vdim >= 1) ? m_vdim : 1.0f;

    // Square
    mat.tex = makeSurfaceTex( 0xffffff, 0, 0x000000, -1, 0x000000, -1, 0.1f, 0.0f );
    m_materials.push_back( mat );
    OTKAppShapeMaker::makeAxisPlane( float3{0, 0, 0}, float3{1, 1, 0}, shape );
    // Fix up texture coordinates for udim size
    for( unsigned int i = 0; i < shape.size(); ++i )
    {
        shape[i].t.x *= udim;
        shape[i].t.y *= vdim;
    }
    addShapeToScene( shape, m_materials.size() - 1 );

    copyGeometryToDevice();
}

void UdimTextureApp::initLaunchParams( OTKAppPerDeviceOptixState& state, unsigned int numDevices )
{
    OTKApp::initLaunchParams( state, numDevices );
    float2* uvdim = reinterpret_cast<float2*>(&state.params.extraData);
    uvdim->x = (m_udim >= 1) ? m_udim : 1.0f;
    uvdim->y = (m_vdim >= 1) ? m_vdim : 1.0f;
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
        "   --texture <texturefile|checkerboard|mandelbrot>    Use texture image file.\n"
        "   --texdim=<width>x<height>   Texture dimensions.\n"
        "   --udim=<width>x<height>     UDIM texture dimensions.\n"
        "   --srgb                      Turn on SRGB conversion for textures.\n"
        "   --dense                     Use dense (standard) textures instead of sparse textures.\n"
        "   --cascade                   Use cascading texture sizes.\n"
        "   --coalesce-tiles            Coalesce white and black tiles.\n"
        "   --coalesce-images           Coalesce identical images.\n"
        "   --dim=<width>x<height>      Specify rendering dimensions.\n"
        "   --file <outputfile>         Render to output file and exit.\n"
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
    const char* textureName  = "mandelbrot";
    const char* outFileName  = "";
    bool        glInterop    = true;

    int  texWidth = 8192;
    int  texHeight = 8192;
    int  udim = 10;
    int  vdim = 10;
    bool useBaseImage = false;
    bool srgb = false;

    demandLoading::Options options{};

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
        else if( arg.substr( 0, 9 ) == "--texdim=" )
            otk::parseDimensions( arg.substr( 9 ).c_str(), texWidth, texHeight );
        else if( arg.substr( 0, 7 ) == "--udim=" )
            otk::parseDimensions( arg.substr( 7 ).c_str(), udim, vdim );
        else if( arg == "--no-gl-interop" )
            glInterop = false;
        else if( arg == "--base-image" )
            useBaseImage = true;
        else if( arg == "--dense" )
            options.useSparseTextures = false;
        else if( arg == "--srgb" )
            srgb = true;
        else if( arg == "--cascade" )
            options.useCascadingTextureSizes = true;
        else if( arg == "--coalesce-tiles" )
            options.coalesceWhiteBlackTiles = true;
        else if( arg == "--coalesce-images" )
            options.coalesceDuplicateImages = true;
        else
            printUsage( argv[0] );
    }

    // Set up options for final frame or interactive rendering.
    bool interactive = strlen( outFileName ) == 0;

    options.maxTexMemPerDevice  = interactive ? 2ULL << 30 : 0ULL;
    options.maxRequestedPages   = interactive ? 64 : 4096;
    options.maxStalePages       = options.maxRequestedPages * 2;
    options.maxInvalidatedPages = options.maxRequestedPages * 2;
    options.maxStagedPages      = options.maxRequestedPages * 2;
    options.maxRequestQueueSize = options.maxRequestedPages * 2;
    options.maxThreads          = 0;

    printKeyCommands();

    // Initialize the app
    UdimTextureApp app( "UDIM Texture Viewer", windowWidth, windowHeight, outFileName, glInterop );
    app.initDemandLoading( options );
    app.setUseSrgb( srgb );
    app.setUdimParams( textureName, texWidth, texHeight, udim, vdim, useBaseImage || ( udim == 0 ) );
    app.createTexture();
    app.createScene();
    app.resetAccumulator();
    app.initOptixPipelines( UdimTextureViewerCudaText(), UdimTextureViewerCudaSize );
    app.startLaunchLoop();
    app.printDemandLoadingStats();

    return 0;
}
