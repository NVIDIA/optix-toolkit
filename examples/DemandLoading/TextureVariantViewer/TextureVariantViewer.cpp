// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "TextureVariantViewerCuda.h"

#include <OptiXToolkit/DemandLoading/TextureSampler.h>
#include <OptiXToolkit/DemandTextureAppBase/DemandTextureApp.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/ImageSources/MultiCheckerImage.h>

#include <assert.h>

using namespace demandTextureApp;
using namespace demandLoading;

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
    std::shared_ptr<imageSource::ImageSource> imageSource = createExrImage( m_textureName );
    if( !imageSource && ( !m_textureName.empty() ) )
        std::cout << "ERROR: Could not find image " << m_textureName << ". Substituting procedural image.\n";
    if( !imageSource )
        imageSource.reset( new imageSources::MultiCheckerImage<float4>( 8192, 8192, 16, true ) );

    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );

        // Make first texture with trilinear filtering.
        demandLoading::TextureDescriptor    texDesc1 = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, FILTER_BILINEAR );
        const demandLoading::DemandTexture& texture1 = state.demandLoader->createTexture( imageSource, texDesc1 );
        if( m_textureIds.size() == 0 )
            m_textureIds.push_back( texture1.getId() );

        // Make second texture with point filtering.
        // The demand loader will share the backing store for textures created with the same imageSource pointer.
        demandLoading::TextureDescriptor    texDesc2 = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, FILTER_POINT );
        const demandLoading::DemandTexture& texture2 = state.demandLoader->createTexture( imageSource, texDesc2 );
        if( m_textureIds.size() == 1 )
            m_textureIds.push_back( texture2.getId() );
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
