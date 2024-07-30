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

#include <chrono>

#include <optix_stubs.h> // This include is needed to avoid a link error
#include <cuda_runtime.h>

#include "CdfInversionParams.h"
#include "CdfInversionKernelCuda.h"

#include <OptiXToolkit/DemandTextureAppBase/DemandTextureApp3D.h>
#include <OptiXToolkit/DemandTextureAppBase/ShapeMaker.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/ImageSources/MultiCheckerImage.h>
#include <OptiXToolkit/ShaderUtil/AliasTable.h>
#include <OptiXToolkit/ShaderUtil/PdfTable.h>
#include <OptiXToolkit/ShaderUtil/CdfInversionTable.h>
#include <OptiXToolkit/ShaderUtil/ISummedAreaTable.h>

#include <OptiXToolkit/Gui/Gui.h>
#include <OptiXToolkit/Gui/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

using namespace demandTextureApp;
using namespace demandLoading;
using namespace imageSource;
using namespace std::chrono;

//------------------------------------------------------------------------------
// Timing
//------------------------------------------------------------------------------

#define TIMEPOINT time_point<high_resolution_clock>
TIMEPOINT now() { return high_resolution_clock::now(); }
double elapsed( TIMEPOINT start ) { return duration_cast<duration<double>>( now() - start ).count(); }

//------------------------------------------------------------------------------
// CdfInversionApp
//------------------------------------------------------------------------------

class CdfInversionApp : public DemandTextureApp3D
{
  public:
    CdfInversionApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop );
    void setTextureName( const char* textureName ) { m_textureName = textureName; }
    void createTexture() override;
    void initView() override;
    void createScene();
    void initLaunchParams( PerDeviceOptixState& state, unsigned int numDevices ) override;
    void setRenderMode( int mode ) { m_render_mode = mode; }
    void setTableMipLevel( int level ) { m_tableMipLevel = level; }
    
    void cleanupState( PerDeviceOptixState& state ) override;
    void drawGui() override;

  protected:
    std::string m_textureName;
    int m_tableMipLevel = 0;
    int m_mipLevel0 = 0;
    int m_numRisSamples = 16;
    std::vector<CdfInversionTable> m_emapInversionTables;
    std::vector<AliasTable> m_emapAliasTables;
    std::vector<ISummedAreaTable> m_emapSummedAreaTables;

    void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods ) override;
    void keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods ) override;
};


CdfInversionApp::CdfInversionApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
    : DemandTextureApp3D( appTitle, width, height, outFileName, glInterop )
{
    m_reset_subframe_threshold = 2;
    m_backgroundColor = float4{1.0f, 1.0f, 1.0f, 0.0f};
    m_projection = Projection::PINHOLE;
    m_lens_width = 0.0f;
    m_render_mode = 1;

    m_emapInversionTables.resize( m_perDeviceOptixStates.size() );
    m_emapAliasTables.resize( m_perDeviceOptixStates.size() );
    m_emapSummedAreaTables.resize( m_perDeviceOptixStates.size() );
}

void CdfInversionApp::initView()
{
    setView( float3{0.0f, 25.0f, 7.0f}, float3{0.0f, 0.0f, 3.0f}, float3{0.0f, 0.0f, 1.0f}, 30.0f );
}

void CdfInversionApp::cleanupState( PerDeviceOptixState& state )
{
    OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
    freeCdfInversionTableDevice( m_emapInversionTables[state.device_idx] );
    freeAliasTableDevice( m_emapAliasTables[state.device_idx] );
    freeISummedAreaTableDevice( m_emapSummedAreaTables[state.device_idx] );
    DemandTextureApp::cleanupState( state );
}

void CdfInversionApp::initLaunchParams( PerDeviceOptixState& state, unsigned int numDevices )
{
    DemandTextureApp::initLaunchParams( state, numDevices );
    state.params.i[SUBFRAME_ID]      = m_subframeId;
    state.params.i[EMAP_ID]          = m_textureIds[0];
    state.params.i[MIP_LEVEL_0_ID]   = m_mipLevel0;
    state.params.i[NUM_RIS_SAMPLES]  = m_numRisSamples; 
    state.params.f[MIP_SCALE_ID]     = m_mipScale;

    CdfInversionTable* cit = reinterpret_cast<CdfInversionTable*>( &state.params.c[EMAP_INVERSION_TABLE_ID] );
    *cit = m_emapInversionTables[ state.device_idx ];
    AliasTable* at = reinterpret_cast<AliasTable*>( &state.params.c[EMAP_ALIAS_TABLE_ID] );
    *at = m_emapAliasTables[ state.device_idx ];
    ISummedAreaTable* sat = reinterpret_cast<ISummedAreaTable*>( &state.params.c[EMAP_SUMMED_AREA_TABLE_ID] );
    *sat = m_emapSummedAreaTables[ state.device_idx ];
}

void CdfInversionApp::createTexture()
{
    // Open the environment map texture
    std::shared_ptr<ImageSource> imageSource( createExrImage( m_textureName.c_str() ) );
    if( !imageSource && !m_textureName.empty() )
        std::cout << "ERROR: Could not find image " << m_textureName << ". Substituting procedural image.\n";
    if( !imageSource )
        imageSource.reset( new imageSources::MultiCheckerImage<half4>( 2048, 1024, 16, true, false ) );
    imageSource::TextureInfo texInfo;
    imageSource->open(&texInfo);

    // Make an environment map texture for each device
    demandLoading::TextureDescriptor texDesc = makeTextureDescriptor( CU_TR_ADDRESS_MODE_WRAP, FILTER_BILINEAR );
    texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        const demandLoading::DemandTexture& texture = state.demandLoader->createTexture( imageSource, texDesc );
        if( m_textureIds.empty() )
            m_textureIds.push_back( texture.getId() );
    }

    // Allocate inversion, alias, and summed area tables on host
    int tableWidth = texInfo.width >> m_tableMipLevel;
    int tableHeight = texInfo.height >> m_tableMipLevel;
    CdfInversionTable hostEmapInversionTable{};
    allocCdfInversionTableHost( hostEmapInversionTable, tableWidth, tableHeight );

    AliasTable hostEmapAliasTable{};
    allocAliasTableHost( hostEmapAliasTable, (int)(tableWidth * tableHeight) );

    ISummedAreaTable hostEmapSummedAreaTable{};
    allocISummedAreaTableHost( hostEmapSummedAreaTable, tableWidth, tableHeight );

    // Read a mip level and make a pdf from it
    TIMEPOINT imageLoadStart = now();
    int bytesPerPixel = getBytesPerChannel(texInfo.format) * texInfo.numChannels;
    char* imgData = (char*)malloc( tableWidth * tableHeight * bytesPerPixel );
    imageSource->readMipLevel( imgData, m_tableMipLevel, tableWidth, tableHeight, CUstream{0} );
    printf( "Time to load image: %0.4f sec.\n", elapsed( imageLoadStart ) );

    TIMEPOINT makePdfStart = now();
    float* pdf = reinterpret_cast<float*>( malloc( tableWidth * tableHeight * sizeof(float) ) );
    if( texInfo.format == CU_AD_FORMAT_UNSIGNED_INT8 )
    {
        makePdfTable<uchar4>( pdf, (uchar4*)imgData, &hostEmapInversionTable.aveValue, 
                              tableWidth, tableHeight, pbLUMINANCE, paLATLONG );
    }
    else if( texInfo.format == CU_AD_FORMAT_HALF )
    {
        makePdfTable<half4>( pdf, (half4*)imgData, &hostEmapInversionTable.aveValue, 
                             tableWidth, tableHeight, pbLUMINANCE, paLATLONG );
    } 
    else if( texInfo.format == CU_AD_FORMAT_FLOAT )
    {
        makePdfTable<float4>( pdf, (float4*)imgData, &hostEmapInversionTable.aveValue, 
                             tableWidth, tableHeight, pbLUMINANCE, paLATLONG );
    }
    printf( "Time to make pdf table: %0.4f sec.\n", elapsed( makePdfStart ) );

    // Make summed area table on host
    TIMEPOINT makeSummedAreaTableStart = now();
    initISummedAreaTable( hostEmapSummedAreaTable, pdf );
    printf( "Time to make summed area table: %0.4f sec.\n", elapsed( makeSummedAreaTableStart ) );

    // Make cdf table on host
    memcpy( hostEmapInversionTable.cdfRows, pdf, tableWidth * tableHeight * sizeof(float) );
    TIMEPOINT makeCdfStart = now();
    invertPdf2D( hostEmapInversionTable );
    printf( "Time to make cdf table: %0.4f sec.\n", elapsed( makeCdfStart ) );

    // Invert cdf table on  host
    TIMEPOINT invertCdfStart = now();
    invertCdf2D( hostEmapInversionTable );
    printf( "Time to invert cdf table: %0.4f sec.\n", elapsed( invertCdfStart ) );

    // Make alias table on host
    TIMEPOINT makeAliasTableStart = now();
    makeAliasTable( hostEmapAliasTable, pdf );
    printf( "Time to make alias table: %0.4f sec.\n", elapsed( makeAliasTableStart ) );

    // Copy tables to devices
    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        bool allocCdf = true;
        #ifdef emDIRECT_LOOKUP
            allocCdf = false;
        #endif

        allocCdfInversionTableDevice( m_emapInversionTables[state.device_idx], tableWidth, tableHeight, allocCdf );
        copyToDevice( hostEmapInversionTable, m_emapInversionTables[state.device_idx] );
        allocAliasTableDevice( m_emapAliasTables[state.device_idx], tableWidth * tableHeight );
        copyToDevice( hostEmapAliasTable, m_emapAliasTables[state.device_idx] );
        allocISummedAreaTableDevice( m_emapSummedAreaTables[state.device_idx], tableWidth, tableHeight );
        copyToDevice( hostEmapSummedAreaTable, m_emapSummedAreaTables[state.device_idx] );
    }

    // Free temp host data structures
    freeAliasTableHost( hostEmapAliasTable );
    freeCdfInversionTableHost( hostEmapInversionTable );
    free( pdf );
    free( imgData );
}

void CdfInversionApp::createScene()
{
    const unsigned int NUM_SEGMENTS = 256;
    TriangleHitGroupData mat{};
    std::vector<Vert> shape;

    // Ground
    mat.tex = makeSurfaceTex( 0x335533, -1, 0x000000, -1, 0x000000, -1, 0.1f, 0.0f );
    m_materials.push_back( mat );
    ShapeMaker::makeAxisPlane( float3{-80, -80, 0}, float3{80, 80, 0}, shape );
    addShapeToScene( shape, m_materials.size() - 1 );

    // ball
    mat.tex = makeSurfaceTex( 0x000000, -1, 0xCCCCCC, -1, 0x000000, -1, 0.00000f, 0.0f );
    m_materials.push_back( mat );
    ShapeMaker::makeSphere( float3{-3.0f, 4.5f, 0.75f}, 0.75f, NUM_SEGMENTS, shape );
    addShapeToScene( shape, m_materials.size() - 1 );

    // Vases
    mat.tex = makeSurfaceTex( 0x773333, -1, 0x010101, -1, 0x000000, -1, 0.02f, 0.0f );
    m_materials.push_back( mat );
    ShapeMaker::makeVase( float3{6.0f, 0.0f, 0.01f}, 0.5f, 2.3f, 8.0f, NUM_SEGMENTS, shape );
    addShapeToScene( shape, m_materials.size() -1 );

    mat.tex = makeSurfaceTex( 0x010101, -1, 0x555566, -1, 0x000000, -1, 0.01f, 0.0f );
    m_materials.push_back( mat );
    ShapeMaker::makeVase( float3{0.0f, 0.0f, 0.01f}, 0.5f, 2.3f, 8.0f, NUM_SEGMENTS, shape );
    addShapeToScene( shape, m_materials.size() -1 );

    mat.tex = makeSurfaceTex( 0x444444, -1, 0x000000, -1, 0x000000, -1, 0.01f, 0.0f );
    m_materials.push_back( mat );
    ShapeMaker::makeVase( float3{-6.0f, 0.0f, 0.01f}, 0.5f, 2.3f, 8.0f, NUM_SEGMENTS, shape );
    addShapeToScene( shape, m_materials.size() -1 );
    
    copyGeometryToDevice();
}

void CdfInversionApp::mouseButtonCallback( GLFWwindow* window, int button, int action, int /*mods*/ )
{
    ImGuiIO& io = ImGui::GetIO();
    io.AddMouseButtonEvent( button, (bool) action );

    if( !io.WantCaptureMouse )
    {
        glfwGetCursorPos( window, &m_mousePrevX, &m_mousePrevY );
        m_mouseButton = ( action == GLFW_PRESS ) ? button : NO_BUTTON;
    }
}

void CdfInversionApp::keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods )
{
    DemandTextureApp::keyCallback( window, key, scancode, action, mods );
    if( action != GLFW_PRESS )
        return;

    if( key == GLFW_KEY_B ) {
        m_mipLevel0 = !m_mipLevel0;
    } else if( key == GLFW_KEY_LEFT ) {
        m_maxSubframes = std::min( 2048, std::max( m_maxSubframes/2, 1 ) );
        printf("max subframes: %d\n", m_maxSubframes);
    } else if( key == GLFW_KEY_RIGHT ) {
        m_maxSubframes = std::min( 2048, std::max( m_maxSubframes*2, 1 ) );
        printf("max subframes: %d\n", m_maxSubframes);
    } else if( key == GLFW_KEY_UP ) {
        m_numRisSamples = std::min( 512, m_numRisSamples*2 );
        printf("RIS samples: %d\n", m_numRisSamples);
    } else if( key == GLFW_KEY_DOWN ) {
        m_numRisSamples = std::max( m_numRisSamples/2, 2);
        printf("RIS samples %d\n", m_numRisSamples);
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

void CdfInversionApp::drawGui()
{
    DemandTextureApp::drawGui();
}

//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------

void printUsage( const char* argv0 )
{
    std::cout << "\n\nUsage: " << argv0 << " [options]\n\n";
    std::cout << "Options:  --scene [0-5], --texture <texturefile.exr>, --launches <numLaunches>\n";
    std::cout << "          --dim=<width>x<height>, --file <outputfile.ppm>, --no-gl-interop\n";
    std::cout << "          --table-mip-level --max-subframes <n>\n\n";
    std::cout << "Mouse:    <LMB>:          pan camera\n";
    std::cout << "          <RMB>:          rotate camera\n\n";
    std::cout << "Keyboard: <ESC>:          exit\n";
    std::cout << "          1-3:            set render mode (1=MIS,2=EMAP, 3=BSDF)\n";
    std::cout << "          <LEFT>,<RIGHT>: change max depth\n";
    std::cout << "          <UP>,<DOWN>:    change min depth\n";
    std::cout << "          WASD,QE:        pan camera\n";
    std::cout << "          J,L:            rotate camera\n";
    std::cout << "          C:              reset view\n";
    std::cout << "          B:              toggle use ray differentials / mip level 0\n";
}

int main( int argc, char* argv[] )
{
    int         windowWidth   = 900;
    int         windowHeight  = 600;
    const char* textureName   = "";
    const char* outFileName   = "";
    bool        glInterop     = true;
    int         numLaunches   = 256;
    int         renderMode    = 0;
    int         tableMipLevel = 0;
    int         maxSubframes  = 1000000;

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
        else if( arg == "--render-mode" && !lastArg )
            renderMode = atoi( argv[++i] ) - 1;
        else if( arg == "--table-mip-level" && !lastArg )
            tableMipLevel = atoi( argv[++i] );
        else if( arg == "--max-subframes" && !lastArg )
            maxSubframes = atoi( argv[++i] );
        else 
            exit(0);
    }

    CdfInversionApp app( "Cdf Inversion", windowWidth, windowHeight, outFileName, glInterop );
    app.initView();
    app.setTableMipLevel( tableMipLevel );
    app.setRenderMode( renderMode );
    app.setNumLaunches( numLaunches );
    app.sceneIsTriangles( true );
    app.initDemandLoading();
    app.setTextureName( textureName );
    app.setMaxSubframes( maxSubframes );
    app.createTexture();
    app.createScene();
    app.resetAccumulator();
    app.initOptixPipelines( CdfInversionCudaText(), CdfInversionCudaSize );
    app.startLaunchLoop();
    app.printDemandLoadingStats();
    
    return 0;
}
