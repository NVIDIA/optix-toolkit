
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

#include <string>

#include <OptiXToolkit/DemandTextureAppBase/DemandTextureApp.h>
#include <OptiXToolkit/ImageSource/MultiCheckerImage.h>
#include <OptiXToolkit/ImageSource/DeviceMandelbrotImage.h>
using namespace demandTextureApp;

extern "C" char UdimTextureViewer_ptx[];  // generated via CMake by embed_ptx.

//------------------------------------------------------------------------------
// UdimTextureApp
// Shows how to create and use udim textures.
//------------------------------------------------------------------------------

class UdimTextureApp : public DemandTextureApp
{
  public:
    UdimTextureApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
        : DemandTextureApp( appTitle, width, height, outFileName, glInterop )
    {
    }
    void setUdimParams( const char* textureName, int texWidth, int texHeight, int udim, int vdim, bool useBaseImage );
    void createTexture() override;

  private:
    std::string m_textureName;
    int         m_texWidth     = 1024;
    int         m_texHeight    = 1024;
    int         m_udim         = 10;
    int         m_vdim         = 10;
    bool        m_useBaseImage = false;
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
    const int           iterations = 512;
    std::vector<float4> colors     = {
        {1.0f, 1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, {0.0f, 0.5f, 0.0f, 0.0f}, {1.0f, 0.0f, .0f, 0.0f}, {1.0f, 1.0f, 0.0f, 0.0f}};

    // Make optional base texture
    int baseTextureId = -1;
    if( m_useBaseImage )
    {
        imageSource::ImageSource* baseImage = nullptr;
        if( m_textureName != "mandelbrot " )
            baseImage = createExrImage( ( m_textureName + ".exr" ).c_str() );
        if( !baseImage )
            baseImage = new imageSource::DeviceMandelbrotImage( m_texWidth, m_texHeight, -2.0, -2.0, 2.0, 2.0, iterations, colors );
        std::unique_ptr<imageSource::ImageSource> baseImageSource( baseImage );

        demandLoading::TextureDescriptor texDesc = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR );
        const demandLoading::DemandTexture& baseTexture = m_demandLoader->createTexture( std::move( baseImageSource ), texDesc );
        baseTextureId                                   = baseTexture.getId();
        if( baseTextureId >= 0 )
            m_textureIds.push_back( baseTextureId );
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
            imageSource::ImageSource* subImage = nullptr;
            if( m_textureName != "mandelbrot" ) // loading exr images
            {
                int         udimNum      = 10000 + v * 100 + u;
                std::string subImageName = m_textureName + std::to_string( udimNum ) + ".exr";
                subImage                 = createExrImage( subImageName.c_str() );
            }
            if( !subImage && m_texWidth == 0 ) // mixing different image sizes
            {
                int maxAspect = 64;
                int w         = std::max( 4 << u, ( 4 << v ) / maxAspect );
                int h         = std::max( 4 << v, ( 4 << u ) / maxAspect );
                subImage      = new imageSource::MultiCheckerImage<float4>( w, h, 4, true );
            }
            if( !subImage ) // many images of the same size
            {
                double xmin = -2.0 + 4.0 * u / m_udim;
                double xmax = -2.0 + 4.0 * ( u + 1.0 ) / m_udim;
                double ymin = -2.0 + 4.0 * v / m_vdim;
                double ymax = -2.0 + 4.0 * ( v + 1.0 ) / m_vdim;
                subImage = new imageSource::DeviceMandelbrotImage( m_texWidth, m_texHeight, xmin, ymin, xmax, ymax, iterations, colors );
            }
            subImageSources.emplace_back( subImage );

            // Note: Use address mode CU_TR_ADDRESS_MODE_BORDER for subimages in tex2DGradUdimBlend calls in OptiX programs.
            // (CU_TR_ADDRESS_MODE_CLAMP for tex2DGradUdim calls).
            subTexDescs.push_back( makeTextureDescriptor( CU_TR_ADDRESS_MODE_BORDER, CU_TR_FILTER_MODE_LINEAR ) );
        }
    }
    const demandLoading::DemandTexture& udimTexture =
        m_demandLoader->createUdimTexture( subImageSources, subTexDescs, m_udim, m_vdim, baseTextureId );

    if( m_textureIds.empty() )
        m_textureIds.push_back( udimTexture.getId() );
}

//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------

void printUsage( const char* argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n\n";
    std::cout << "Options:  --texture <mandelbrot|texturefile.exr>, --dim=<width>x<height>, --file <outputfile.ppm>\n";
    std::cout << "          --no-gl-interop, --texdim=<width>x<height>, --udim=<udim>x<vdim>, --base-image\n";
    std::cout << "Keyboard: <ESC>:exit, WASD:pan, QE:zoom, C:recenter\n";
    std::cout << "Mouse:    <LMB>:pan, <RMB>:zoom\n" << std::endl;
    exit(0);
}

int main( int argc, char* argv[] )
{
    int         windowWidth  = 768;
    int         windowHeight = 768;
    const char* textureName  = "mandelbrot";
    const char* outFileName  = "";
    bool        glInterop    = true;

    int texWidth = 8192;
    int texHeight = 8192;
    int udim = 10;
    int vdim = 10;
    bool useBaseImage = false;

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
        else
            printUsage( argv[0] );
    }

    UdimTextureApp app( "UDIM Texture Viewer", windowWidth, windowHeight, outFileName, glInterop );
    app.initDemandLoading();
    app.setUdimParams( textureName, texWidth, texHeight, udim, vdim, useBaseImage || ( udim == 0 ) );
    app.createTexture();
    app.initOptixPipelines( UdimTextureViewer_ptx );
    app.startLaunchLoop();
    app.printDemandLoadingStats();

    return 0;
}