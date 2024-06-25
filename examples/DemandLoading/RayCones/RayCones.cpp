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

#include <OptiXToolkit/DemandTextureAppBase/DemandTextureApp3D.h>
#include <OptiXToolkit/DemandTextureAppBase/ShapeMaker.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>
#include <OptiXToolkit/ImageSources/MultiCheckerImage.h>
#include <OptiXToolkit/ShaderUtil/ray_cone.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include "RayConesParams.h"
#include "RayConesKernelCuda.h"

using namespace otk;
using namespace demandTextureApp;
using namespace demandLoading;
using namespace imageSource;

//------------------------------------------------------------------------------
// RayConesApp
//------------------------------------------------------------------------------

class RayConesApp : public DemandTextureApp3D
{
  public:
    RayConesApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop );
    void setTextureName( const char* textureName ) { m_textureName = textureName; }
    void createTexture() override;
    void initView() override;
    void createScene();
    void setSceneId( int sceneId ) { m_sceneId = sceneId; }
    void initLaunchParams( PerDeviceOptixState& state, unsigned int numDevices ) override;
   
  protected:
    std::string m_textureName;
    int m_sceneId = 0;

    int m_minRayDepth = 0;
    int m_maxRayDepth = 6;
    int m_updateRayCones = 1;

    void keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods ) override; 
};


RayConesApp::RayConesApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
    : DemandTextureApp3D( appTitle, width, height, outFileName, glInterop )
{
    m_backgroundColor = float4{1.0f, 1.0f, 1.0f, 0.0f};
    m_projection = Projection::THINLENS;
    m_lens_width = 0.1f;
    m_render_mode = 1;
}

void RayConesApp::initView()
{
    if( m_sceneId < 5 )
        setView( float3{0.0f, 25.0f, 7.0f}, float3{0.0f, 0.0f, 3.0f}, float3{0.0f, 0.0f, 1.0f}, 30.0f );
    else
        setView( float3{-8.0f, 0.0f, 15.0f}, float3{0.0f, 0.0f, 0.0f}, float3{0.0f, 1.0f, 0.0f}, 30.0f );
}

void RayConesApp::initLaunchParams( PerDeviceOptixState& state, unsigned int numDevices )
{
    DemandTextureApp::initLaunchParams( state, numDevices );
    state.params.i[MIN_RAY_DEPTH_ID]    = m_minRayDepth;
    state.params.i[MAX_RAY_DEPTH_ID]    = m_maxRayDepth;
    state.params.i[SUBFRAME_ID]         = m_subframeId;
    state.params.i[UPDATE_RAY_CONES_ID] = m_updateRayCones;
    state.params.f[MIP_SCALE_ID]        = m_mipScale;
}

void RayConesApp::createTexture()
{
    std::shared_ptr<ImageSource> imageSource( createExrImage( m_textureName ) );
    if( !imageSource && !m_textureName.empty() )
        std::cout << "ERROR: Could not find image " << m_textureName << ". Substituting procedural image.\n";
    if( !imageSource )
        imageSource.reset( new imageSources::MultiCheckerImage<uchar4>( 16384, 16384, 64, true ) );
    
    demandLoading::TextureDescriptor texDesc = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, FILTER_BILINEAR );

    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        const demandLoading::DemandTexture& texture = state.demandLoader->createTexture( imageSource, texDesc );
        if( m_textureIds.empty() )
            m_textureIds.push_back( texture.getId() );
    }
}

void RayConesApp::createScene()
{
    const unsigned int NUM_SEGMENTS = 128;
    TriangleHitGroupData mat{};
    std::vector<Vert> shape;

    // ground plane
    if( m_sceneId == 0 )
    {
        // Ground
        mat.tex = makeSurfaceTex( 0xeeeeee, 0, 0x010101, -1, 0x000000, -1, 0.1f, 0.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeAxisPlane( float3{-80, -80, 0}, float3{80, 80, 0}, shape );
        addShapeToScene( shape, m_materials.size() - 1 );
    }

    // glass and steel balls
    if( m_sceneId == 1 )
    {
        // Ground
        mat.tex = makeSurfaceTex( 0xeeeeee, 0, 0x010101, -1, 0x000000, -1, 0.1f, 0.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeAxisPlane( float3{-80, -80, 0}, float3{80, 80, 0}, shape );
        addShapeToScene( shape, m_materials.size() - 1 );

        // balls
        mat.tex = makeSurfaceTex( 0x000000, -1, 0xffffff, -1, 0x000000, -1, 0.0, 10.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeSphere( float3{4.0f, 0.0f, 3.5f}, 3.5f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() - 1 );

        mat.tex = makeSurfaceTex( 0x000000, -1, 0xeeeeee, -1, 0xeeeeee, -1, 0.0, 1.5f );
        m_materials.push_back( mat );
        ShapeMaker::makeSphere( float3{-4.0f, 0.0f, 3.5f}, 3.5f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() - 1 );
    }

    // vases and spheres
    if( m_sceneId == 2 )
    {
        // Ground
        mat.tex = makeSurfaceTex( 0x777777, -1, 0x777777, -1, 0x000000, -1, 0.01, 0.0f );
        m_materials.push_back( mat );

        ShapeMaker::makeAxisPlane( float3{-40, -40, 0}, float3{40, 40, 0}, shape );
        addShapeToScene( shape, m_materials.size() - 1 );

        // Vases
        mat.tex = makeSurfaceTex( 0xffffff, 0, 0x252525, -1, 0x000000, -1, 0.0001, 0.0f );
        m_materials.push_back( mat );

        ShapeMaker::makeVase( float3{7.0f, 0.0f, 0.01f}, 0.5f, 2.3f, 4.5f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() -1 );
        ShapeMaker::makeVase( float3{0.0f, 0.0f, 0.01f}, 1.0f, 4.0f, 8.0f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() -1 );
        ShapeMaker::makeVase( float3{-7.0f, 0.0f, 0.01f}, 0.5f, 1.5f, 6.0f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() -1 );

        // Vase liners with diffuse material to block negative curvature traps
        mat.tex = makeSurfaceTex( 0x111111, -1, 0x111111, -1, 0x000000, -1, 0.1, 0.0f );
        m_materials.push_back( mat );

        ShapeMaker::makeVase( float3{7.0f, 0.0f, 0.01f}, 0.49f, 2.29f, 4.5f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() -1 );
        ShapeMaker::makeVase( float3{0.0f, 0.0f, 0.01f}, 0.99f, 3.99f, 8.0f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() -1 );
        ShapeMaker::makeVase( float3{-7.0f, 0.0f, 0.01f}, 0.49f, 1.49f, 6.0f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() -1 );

        // Spheres
        mat.tex = makeSurfaceTex( 0x000000, -1, 0xffffff, -1, 0xffffff, -1, 0.0, 1.5f );
        m_materials.push_back( mat );

        ShapeMaker::makeSphere( float3{-5.0f, 1.0f, 0.7f}, 0.7f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() - 1 );

        ShapeMaker::makeSphere( float3{1.0f, 7.0f, 3.7f}, 2.0f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() - 1 );
    }

    // spheres with differing roughness
    if( m_sceneId == 3 )
    {
        // Ground
        mat.tex = makeSurfaceTex( 0xeeeeee, 0, 0x000001, -1, 0x000000, -1, 0.1f, 0.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeAxisPlane( float3{-80, -80, 0}, float3{80, 80, 0}, shape );
        addShapeToScene( shape, m_materials.size() - 1 );

        // sphere
        mat.tex = makeSurfaceTex( 0x000000, -1, 0xffffff, -1, 0x000000, -1, 0.001f, 10.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeSphere( float3{5.5f, 0.0f, 3.5f}, 2.4f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() - 1 );

        // sphere
        mat.tex = makeSurfaceTex( 0x000000, -1, 0xffffff, -1, 0x000000, -1, 0.01, 10.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeSphere( float3{0.0f, 0.0f, 3.5f}, 2.4f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() - 1 );

        // sphere
        mat.tex = makeSurfaceTex( 0x000000, -1, 0xffffff, -1, 0x000000, -1, 0.1, 10.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeSphere( float3{-5.5f, 0.0f, 3.5f}, 2.4f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() - 1 );
    }

    // cylinder
    if( m_sceneId == 4 )
    {
        // Ground
        mat.tex = makeSurfaceTex( 0xeeeeee, 0, 0x010101, -1, 0x000000, -1, 0.1f, 0.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeAxisPlane( float3{-80, -80, 0}, float3{80, 80, 0}, shape );
        addShapeToScene( shape, m_materials.size() - 1 );

        // cylinder
        mat.tex = makeSurfaceTex( 0x000000, -1, 0x555555, -1, 0x000000, -1, 0.0, 0.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeCylinder( float3{0.0, -5.0f, 0.0f}, 7.5f, 2.5f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() - 1 );

        mat.tex = makeSurfaceTex( 0x555555, -1, 0x333333, -1, 0x000000, -1, 0.01, 0.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeCylinder( float3{0.0, -5.0f, 0.0f}, 7.51f, 2.5f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() - 1 );
    }

    // concave reflector
    if( m_sceneId >= 5 )
    {
        // Ground
        mat.tex = makeSurfaceTex( 0x222222, -1, 0x000000, -1, 0x000000, -1, 0.1, 0.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeAxisPlane( float3{-80, -80, 0}, float3{80, 80, 0}, shape );
        addShapeToScene( shape, m_materials.size() - 1 );

        mat.tex = makeSurfaceTex( 0xeeeeee, 0, 0x000000, -1, 0x000000, -1, 0.1, 0.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeSphere( float3{2.0f, 0.0f, 4.5f}, 1.0f, NUM_SEGMENTS, shape );
        addShapeToScene( shape, m_materials.size() - 1 );

        // reflector
        mat.tex = makeSurfaceTex( 0x000000, -1, 0xeeeeee, -1, 0x000000, -1, 0.0, 0.0f );
        m_materials.push_back( mat );
        ShapeMaker::makeSphere( float3{0.0f, 0.0f, 7.5f}, 7.5f, NUM_SEGMENTS, shape, 0.0f, 0.55f );
        addShapeToScene( shape, m_materials.size() - 1 );
    }

    copyGeometryToDevice();
}


void RayConesApp::keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods )
{
    DemandTextureApp3D::keyCallback( window, key, scancode, action, mods );
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
    }

    m_subframeId = 0;
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

    RayConesApp app( "Ray Cones", windowWidth, windowHeight, outFileName, glInterop );
    app.setSceneId( sceneId );
    app.initView();
    app.setNumLaunches( numLaunches );
    app.sceneIsTriangles( true );
    app.initDemandLoading();
    app.setTextureName( textureName );
    app.createTexture();
    app.createScene();
    app.resetAccumulator();
    app.initOptixPipelines( RayConesCudaText(), RayConesCudaSize );
    app.startLaunchLoop();
    app.printDemandLoadingStats();
    
    return 0;
}
