
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

#include <DemandTextureViewerKernelCuda.h>

#include <OptiXToolkit/DemandTextureAppBase/DemandTextureApp.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/ImageSource/MipMapImageSource.h>
#include <OptiXToolkit/ImageSource/TiledImageSource.h>
#include <OptiXToolkit/ImageSources/DeviceMandelbrotImage.h>
#include <OptiXToolkit/ImageSources/ImageSources.h>
#include <OptiXToolkit/ImageSources/MultiCheckerImage.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <memory>
#include <stdexcept>

using namespace demandTextureApp;
using namespace demandLoading;

//------------------------------------------------------------------------------
// DemandTextureViewer
// Shows basic use of OptiX demand textures.
//------------------------------------------------------------------------------

using ImageSourcePtr = std::shared_ptr<imageSource::ImageSource>;

class DemandTextureViewer : public DemandTextureApp
{
  public:
    enum TextureType
    {
        TEXTURE_NONE = 0,
        TEXTURE_FILE,
        TEXTURE_CHECKERBOARD,
        TEXTURE_MANDELBROT
    };

    DemandTextureViewer( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop, bool tile, bool mipmap )
        : DemandTextureApp( appTitle, width, height, outFileName, glInterop )
        , m_tile( tile )
        , m_mipmap( mipmap )
    {
    }

    void           setTextureType( TextureType textureType ) { m_textureType = textureType; }
    void           setTextureName( const std::string& textureName ) { m_textureName = textureName; }
    ImageSourcePtr createImageSource();
    void           createTexture() override;

  private:
    TextureType         m_textureType{};
    std::string         m_textureName;
    std::vector<float4> m_colorMap;
    bool                m_tile{};
    bool                m_mipmap{};
};

float4 hsva( float hue, float saturation, float value, float /*alpha*/ )
{
    using namespace otk;  // for vec_math operators
    const float c = value * saturation;
    const float x = c * std::fabs( std::fmod( ( hue / 60.0f ), 2.0f ) - 1.0f );
    const float m = value - c;
    float3      result;
    if( hue >= 0.0f && hue < 60.0f )
    {
        result = make_float3( c, x, 0 );
    }
    else if( hue >= 60.0f && hue < 120.0f )
    {
        result = make_float3( x, c, 0 );
    }
    else if( hue >= 120.0f && hue < 180.0f )
    {
        result = make_float3( 0, c, x );
    }
    else if( hue >= 180.0f && hue < 240.0f )
    {
        result = make_float3( 0, x, c );
    }
    else if( hue >= 240.0f && hue < 300.0f )
    {
        result = make_float3( x, 0, c );
    }
    else  // >= 300 && < 360
    {
        result = make_float3( c, 0, x );
    }
    result += make_float3( m, m, m );

    return make_float4( result.x, result.y, result.z, 1.0f );
}

static std::vector<float4> createColorMap( int numColors )
{
    std::vector<float4> colorMap( numColors );
    const float         hueStep{360.0f / ( numColors - 1 )};
    float               hue{};
    for( int i = 0; i < numColors; ++i )
    {
        colorMap[i] = hsva( hue, 1.0f, 1.0f, 1.0f );
        hue += hueStep;
    }
    return colorMap;
}

static std::string toString( DemandTextureViewer::TextureType textureType )
{
    switch( textureType )
    {
        case DemandTextureViewer::TEXTURE_NONE:
            return "none";
        case DemandTextureViewer::TEXTURE_FILE:
            return "file";
        case DemandTextureViewer::TEXTURE_CHECKERBOARD:
            return "checkerboard";
        case DemandTextureViewer::TEXTURE_MANDELBROT:
            return "mandelbrot";
    }
    return "unknown";
}

inline bool endsWith( const std::string& text, const std::string& suffix )
{
    return text.length() >= suffix.length() && text.substr( text.length() - suffix.length() ) == suffix;
}

ImageSourcePtr DemandTextureViewer::createImageSource()
{
    if( m_textureType == TEXTURE_NONE )
        m_textureType = TEXTURE_CHECKERBOARD;

    ImageSourcePtr img;
    if( m_textureType == TEXTURE_FILE )
    {
        // Assume EXR images are tiled and mipmapped.
        if( endsWith( m_textureName, ".exr" ) )
            img = createExrImage( m_textureName );
        else
        {
            img = imageSources::createImageSource( m_textureName );
            if( m_mipmap )
            {
                img = createMipMapImageSource( img );
            }
            if( m_tile )
            {
                img = createTiledImageSource( img );
            }
        }
        if( !img || m_textureName.empty() )
        {
            std::cout << "ERROR: Could not find image '" << m_textureName << "'. Substituting procedural image.\n";
            m_textureType = TEXTURE_CHECKERBOARD;
        }
    }
    if( m_textureType == TEXTURE_CHECKERBOARD )
    {
        img = std::make_shared<imageSources::MultiCheckerImage<float4>>( 8192, 8192, 16, true );
    }
    else if( m_textureType == TEXTURE_MANDELBROT )
    {
        const int MAX_ITER = 256;
        m_colorMap         = createColorMap( imageSources::MAX_MANDELBROT_COLORS );
        img = std::make_shared<imageSources::DeviceMandelbrotImage>( 8192, 8192, -2.0, -1.5, 1, 1.5, MAX_ITER, m_colorMap );
    }
    if( !img )
    {
        throw std::runtime_error( "Could not create requested texture " + toString( m_textureType )
                                  + ( m_textureName.empty() ? std::string{} : " (" + m_textureName + ")" ) );
    }
    return img;
}

void DemandTextureViewer::createTexture()
{
    ImageSourcePtr imageSource( createImageSource() );

    demandLoading::TextureDescriptor texDesc = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, FILTER_BILINEAR );
    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        const demandLoading::DemandTexture& texture = state.demandLoader->createTexture( imageSource, texDesc );
        if( m_textureIds.empty() )
            m_textureIds.push_back( texture.getId() );
    }
}

//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------

void printUsage( const char* program )
{
    // clang-format off
    std::cerr << "\nUsage: " << program << " [options]\n"
        "\n"
        "Options:\n"
        "   --texture <texturefile.exr>     Use texture image file.\n"
        "   --checkerboard                  Use procedural checkerboard texture.\n"
        "   --mandelbrot                    Use procedural Mandelbrot texture.\n"
        "   --dim=<width>x<height>          Specify rendering dimensions.\n"
        "   --file <outputfile>             Render to output file and exit.\n"
        "   --tile                          Make image tileable\n"
        "   --mipmap                        Make image mipmapped\n"
        "   --no-gl-interop                 Disable OpenGL interop.\n";
    // clang-format on

    exit(0);
}

int main( int argc, char* argv[] )
{
    int                              windowWidth  = 768;
    int                              windowHeight = 768;
    std::string                      textureName;
    std::string                      outFileName;
    bool                             glInterop = true;
    bool                             tile{ false };
    bool                             mipmap{ false };
    DemandTextureViewer::TextureType textureType{DemandTextureViewer::TEXTURE_NONE};

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        const bool        lastArg = ( i == argc - 1 );

        if( arg == "--texture" && !lastArg )
        {
            textureName = argv[++i];
            textureType = DemandTextureViewer::TEXTURE_FILE;
        }
        else if( arg == "--checkerboard" )
            textureType = DemandTextureViewer::TEXTURE_CHECKERBOARD;
        else if( arg == "--mandelbrot" )
            textureType = DemandTextureViewer::TEXTURE_MANDELBROT;
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
        else
            printUsage( argv[0] );
    }

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

    DemandTextureViewer app( "Texture Viewer", windowWidth, windowHeight, outFileName, glInterop, tile, mipmap );
    app.initDemandLoading();
    app.setTextureType( textureType );
    app.setTextureName( textureName );
    app.createTexture();
    app.initOptixPipelines( DemandTextureViewerCudaText(), DemandTextureViewerCudaSize );
    app.startLaunchLoop();
    app.printDemandLoadingStats();

    return 0;
}
