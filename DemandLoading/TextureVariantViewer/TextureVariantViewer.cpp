
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

#include <assert.h>

#include <TextureVariantViewerCuda.h>

#include <OptiXToolkit/DemandLoading/TextureSampler.h>
#include <OptiXToolkit/DemandTextureAppBase/DemandTextureApp.h>
#include <OptiXToolkit/ImageSources/MultiCheckerImage.h>

using namespace demandLoading;
using namespace demandTextureApp;

//------------------------------------------------------------------------------
// TextureVariantApp
// Shows the use of texture variants (multiple textures with the same backing store).
//------------------------------------------------------------------------------

class TextureVariantApp : public DemandTextureApp
{
  public:
    TextureVariantApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
        : DemandTextureApp( appTitle, width, height, outFileName, glInterop )
    {
    }
    void setTextureName( const char* textureName ) { m_textureName = textureName; };
    void createTexture() override;

  private:
    std::string m_textureName;
};

void TextureVariantApp::createTexture()
{
    imageSource::ImageSource* img =  createExrImage( m_textureName.c_str() );
    if( !img && ( !m_textureName.empty() ) )
        std::cout << "ERROR: Could not find image " << m_textureName << ". Substituting procedural image.\n";
    if( !img )
        img = new imageSources::MultiCheckerImage<float4>( 8192, 8192, 16, true );
    
    std::shared_ptr<imageSource::ImageSource> imageSource( img );

    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );

        // Make first texture with trilinear filtering.
        demandLoading::TextureDescriptor    texDesc1 = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR );
        const demandLoading::DemandTexture& texture1 = state.demandLoader->createTexture( imageSource, texDesc1 );
        if( m_textureIds.size() == 0 )
            m_textureIds.push_back( texture1.getId() );

        // Make second texture with point filtering.
        // The demand loader will share the backing store for textures created with the same imageSource pointer.
        demandLoading::TextureDescriptor    texDesc2 = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_POINT );
        const demandLoading::DemandTexture& texture2 = state.demandLoader->createTexture( imageSource, texDesc2 );
        if( m_textureIds.size() == 1 )
            m_textureIds.push_back( texture2.getId() );

        // The OptiX closest hit program assumes the textures are consecutive.
        assert( texture2.getId() == texture1.getId() + PAGES_PER_TEXTURE ); 
    }
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

void printUsage( const char* argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n\n";
    std::cout << "Options:  --texture <texturefile.exr>, --dim=<width>x<height>, --file <outputfile.ppm> --no-gl-interop\n";
    std::cout << "Keyboard: <ESC>:exit, WASD:pan, QE:zoom, C:recenter\n";
    std::cout << "Mouse:    <LMB>:pan, <RMB>:zoom\n" << std::endl;
    exit(0);
}

int main( int argc, char* argv[] )
{
    int         windowWidth  = 768;
    int         windowHeight = 768;
    const char* textureName  = "";
    const char* outFileName  = "";
    bool        glInterop    = true;

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
        else
            printUsage( argv[0] );
    }

    TextureVariantApp app( "Texture Variants. Trilinear (left) and point filtering (right).", 
                           windowWidth, windowHeight, outFileName, glInterop );
    app.initDemandLoading();
    app.setTextureName( textureName );
    app.createTexture();
    app.initOptixPipelines( TextureVariantViewerCudaText(), TextureVariantViewerCudaSize );
    app.startLaunchLoop();
    app.printDemandLoadingStats();
    
    return 0;
}
