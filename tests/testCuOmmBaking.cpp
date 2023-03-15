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

#include "SourceDir.h"  // generated from SourceDir.h.in

#include <OptiXToolkit/CuOmmBaking/CuBuffer.h>
#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>

#include "Util/Exception.h"
#include "Util/Image.h"
#include "Util/Mesh.h"
#include "Util/OptiXOmmArray.h"
#include "Util/OptiXScene.h"
#include "Util/BakeTexture.h"

#include "testCommon.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {  // anonymous

class OmmBakingTest : public TestCommon
{
protected:

    struct TestOptions
    {
        struct MeshOptions
        {
            Mesh::Format format;
            float3 vertMin = { 0, 0, 0 }, vertMax = { 1, 1, 0 };
            float2 uvMin = { 0, 0 }, uvMax = { 1, 1 };
            uint2  meshResolution = { 5, 5 };  // 5x5 tiles

            const float* transform = 0;

            std::vector<unsigned> textures = { 0 };
        };

        struct Texture
        {
            cudaTextureAddressMode            addressMode[2] = {};
            cudaTextureReadMode               readMode = cudaReadModeNormalizedFloat;
            cudaTextureFilterMode             filterMode = cudaFilterModePoint;
            struct cudaChannelFormatDesc      desc = cudaCreateChannelDesc<uchar4>();
            cuOmmBaking::CudaTextureAlphaMode alphaMode = cuOmmBaking::CudaTextureAlphaMode::DEFAULT;

            std::string texture = { getSourceDir() + "/Textures/base/base.png" };

            float transparencyCutoff = 0.f;
            float opacityCutoff = 1.f;

            bool bakeToState = false;
            uint32_t statePaddingInBits = 0;
        };

        cuOmmBaking::BakeOptions options;

        float filterKernelWidthInTexels = 0.f;

        std::vector<MeshOptions> meshes   = { MeshOptions{} };
        std::vector<Texture>     textures = { Texture{} };

        int layer = 0;
    };

    OptixOmmArray ommArray = {};

    std::vector<Mesh>               meshes;
    std::vector<CuTexture>          textures;
    std::vector<CuBuffer<uint32_t>> bakedTextures;

    std::vector<std::vector<cuOmmBaking::TextureDesc>> inputTextures;
    std::vector<cuOmmBaking::BakeInputDesc> bakeInputs;
    
    

    CuBuffer<float> d_preTransform;
    
    cuOmmBaking::Result buildOmm( const TestOptions& opt )
    {
        // Create scene geometry
        
        std::vector<cuOmmBaking::TextureDesc> cudaTextureInputs;
        std::vector<cuOmmBaking::TextureDesc> textureInputs;

        bakeInputs.clear();
        inputTextures.clear();
        bakedTextures.clear();

        meshes.resize( opt.meshes.size() );

        for( size_t i = 0; i < opt.meshes.size(); ++i )
        {
            OMM_CUDA_CHECK( meshes[i].create( opt.meshes[i].vertMin, opt.meshes[i].vertMax, opt.meshes[i].uvMin, opt.meshes[i].uvMax, opt.meshes[i].meshResolution, opt.meshes[i].textures.size(), opt.meshes[i].format));
        }

        textures.resize( opt.textures.size() );
        bakedTextures.resize( opt.textures.size() );

        // Build memory texture
        for( size_t i = 0; i < opt.textures.size(); ++i )
        {
            struct cudaTextureDesc texDesc;
            memset( &texDesc, 0, sizeof( texDesc ) );
            switch( opt.textures[i].desc.f )
            {
            case cudaChannelFormatKindFloat:
                texDesc.readMode = cudaReadModeElementType;
                break;
            default:
                texDesc.readMode = opt.textures[i].readMode;
                break;
            }
            texDesc.filterMode = opt.textures[i].filterMode;
            texDesc.normalizedCoords = 1;
            texDesc.addressMode[0] = opt.textures[i].addressMode[0];
            texDesc.addressMode[1] = opt.textures[i].addressMode[1];

            OMM_CUDA_CHECK( textures[i].createFromFile(opt.textures[i].texture.c_str(), 0, &texDesc, &opt.textures[i].desc));

            cuOmmBaking::TextureDesc cudaTexDesc = {};
            cudaTexDesc.type = cuOmmBaking::TextureType::CUDA;
            cudaTexDesc.cuda.transparencyCutoff = opt.textures[i].transparencyCutoff;
            cudaTexDesc.cuda.opacityCutoff = opt.textures[i].opacityCutoff;
            cudaTexDesc.cuda.texObject = textures[i].getTexture();
            cudaTexDesc.cuda.filterKernelWidthInTexels = opt.filterKernelWidthInTexels;
            cudaTexDesc.cuda.alphaMode = opt.textures[i].alphaMode;

            if( opt.textures[i].bakeToState )
            {
                cudaChannelFormatDesc chanDesc = {};
                cudaResourceDesc      resDesc = {};
                cudaExtent            extent = {};

                cudaTextureObject_t texObject = textures[i].getTexture();
                OMM_CUDA_CHECK( cudaGetTextureObjectResourceDesc( &resDesc, texObject ) );

                switch( resDesc.resType )
                {
                case cudaResourceTypeArray: {
                    OMM_CUDA_CHECK( cudaGetChannelDesc( &chanDesc, resDesc.res.array.array ) );
                    OMM_CUDA_CHECK( cudaArrayGetInfo( 0, &extent, 0, resDesc.res.array.array ) );
                } break;
                case cudaResourceTypeMipmappedArray: {
                    cudaArray_t d_topLevelArray;
                    OMM_CUDA_CHECK( cudaGetMipmappedArrayLevel( &d_topLevelArray, resDesc.res.mipmap.mipmap, 0 ) );
                    OMM_CUDA_CHECK( cudaGetChannelDesc( &chanDesc, d_topLevelArray ) );
                    OMM_CUDA_CHECK( cudaArrayGetInfo( 0, &extent, 0, d_topLevelArray ) );
                } break;
                case cudaResourceTypePitch2D: {
                    extent.width = resDesc.res.pitch2D.width;
                    extent.height = resDesc.res.pitch2D.height;
                    chanDesc = resDesc.res.pitch2D.desc;
                } break;
                default:
                    return cuOmmBaking::Result::ERROR_INVALID_VALUE;
                };

                size_t pitchInBits = extent.width * 2 + opt.textures[i].statePaddingInBits;

                bakedTextures[i].alloc( ( ( pitchInBits * extent.height ) + 31 ) / 32 );
                bakedTextures[i].set();

                TextureToStateParams params = {};

                params.tex = texObject;
                params.opacityCutoff = opt.textures[i].opacityCutoff;
                params.transparencyCutoff = opt.textures[i].transparencyCutoff;
                params.isNormalizedCoords = texDesc.normalizedCoords;
                params.isRgba = ( chanDesc.w );
                params.width = extent.width;
                params.height = extent.height;
                params.pitchInBits = pitchInBits;
                params.buffer = (uint32_t*)bakedTextures[i].get();
                
                OMM_CUDA_CHECK( launchTextureToState( params, 0 ) );

                cuOmmBaking::TextureDesc stateTexDesc = {};
                stateTexDesc.type = cuOmmBaking::TextureType::STATE;
                stateTexDesc.state.width = extent.width;
                stateTexDesc.state.height = extent.height;
                stateTexDesc.state.pitchInBits = pitchInBits;
                stateTexDesc.state.addressMode[0] = texDesc.addressMode[0];
                stateTexDesc.state.addressMode[1] = texDesc.addressMode[1];
                stateTexDesc.state.stateBuffer = bakedTextures[i].get();
                stateTexDesc.state.filterKernelWidthInTexels = ( opt.filterKernelWidthInTexels == 0.f ) ? 1.f : opt.filterKernelWidthInTexels;
                textureInputs.push_back( stateTexDesc );
            }
            else
            {
                textureInputs.push_back( cudaTexDesc );
            }

            cudaTextureInputs.push_back( cudaTexDesc );
        }

        std::vector<float> h_preTransform;
        h_preTransform.resize( 6 * opt.meshes.size() );
        d_preTransform.alloc( 6 * opt.meshes.size() );

        inputTextures.resize( opt.meshes.size() );
        for( size_t i = 0; i < opt.meshes.size(); ++i )
        {
            for( const unsigned textureId : opt.meshes[i].textures )
                inputTextures[i].push_back( textureInputs[textureId] );

            cuOmmBaking::BakeInputDesc desc = meshes[i].getBakingInputDesc();
            desc.numTextures = inputTextures[i].size();
            desc.textures    = inputTextures[i].data();
            if( opt.meshes[i].transform )
            {
                memcpy( h_preTransform.data() + 6*i, opt.meshes[i].transform, sizeof(float) * 6);

                desc.transformFormat = cuOmmBaking::UVTransformFormat::MATRIX_FLOAT2X3;
                desc.transform    = d_preTransform.get( 6*i );
            }

            bakeInputs.push_back( desc );
        }
        d_preTransform.upload( h_preTransform.data() );

        // Create the optix omm array
        OMM_CHECK( ommArray.create( optixContext, opt.options, bakeInputs.data(), (uint32_t)bakeInputs .size() ) );

        // Overwrite texture inputs with their cuda equivalents as the optix renderer can't render state inputs directly.
        for( size_t i = 0; i < opt.meshes.size(); ++i )
            for( size_t j = 0; j < opt.meshes[i].textures.size(); ++j )
                inputTextures[i][j] = cudaTextureInputs[opt.meshes[i].textures[j]];

        return cuOmmBaking::Result::SUCCESS;
    }

    cuOmmBaking::Result renderOmm( const TestOptions& opt, const std::string& imageNamePrefix )
    {
        // Build optix scene corresponding to the omm build inputs
        OptixOmmScene scene;
        EXPECT_EQ( OPTIX_SUCCESS, scene.build( optixContext, 0, 0, ommArray, bakeInputs.data(), ( uint32_t )bakeInputs.size() ) );

        // Render a frame

        RenderOptions renderOptions = {};
        renderOptions.opacity_shading = true;
        renderOptions.validate_opacity = true;
        renderOptions.textureLayer = opt.layer;
        renderOptions.windowMin = { FLT_MAX,FLT_MAX };
        renderOptions.windowMax = { FLT_MIN,FLT_MIN };
        renderOptions.force2state = ( opt.options.format == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE );

        for( const auto& mesh : opt.meshes )
        {
            renderOptions.windowMin.x = std::min( mesh.uvMin.x, renderOptions.windowMin.x );
            renderOptions.windowMin.y = std::min( mesh.uvMin.y, renderOptions.windowMin.y );

            renderOptions.windowMax.x = std::max( mesh.uvMax.x, renderOptions.windowMax.x );
            renderOptions.windowMax.y = std::max( mesh.uvMax.y, renderOptions.windowMax.y );
        }

        const int width = 1024;
        const int height = 1024;
        EXPECT_EQ( OPTIX_SUCCESS, scene.render( width, height, renderOptions ) );

        // validate that the omms where conservative.
        EXPECT_EQ( scene.getErrorCount(), 0u );

        // Save the frame to file

        const std::vector<uchar3>& image = scene.getImage();

        OMM_CHECK( saveImageToFile( imageNamePrefix, image, width, height ) );

        scene.destroy();

        return cuOmmBaking::Result::SUCCESS;
    }

    cuOmmBaking::Result runTest( const TestOptions& opt, const std::string& imageNamePrefix )
    {
        OMM_CHECK( buildOmm( opt ) );

        OMM_CHECK( renderOmm( opt, imageNamePrefix ) );

        return cuOmmBaking::Result::SUCCESS;
    }
};

}  // namespace

TEST_F( OmmBakingTest, Base )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.png" ;
    opt.meshes[0].meshResolution = {1, 1};

    cuOmmBaking::Result res = runTest( opt, "OmmBakingTest_Base" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    EXPECT_GE( ommArray.getNumOmms(), 1u );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

class OmmBakingStateTexture : public OmmBakingTest {};

TEST_F( OmmBakingStateTexture, Base )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.png" ;
    opt.textures[0].bakeToState = true;
    opt.meshes[0].meshResolution = { 1, 1 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingStateTexture_Base" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    EXPECT_GE( ommArray.getNumOmms(), 1u );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingStateTexture, Mixed )
{
    TestOptions opt = {};
    opt.textures[0].bakeToState = true;
    opt.textures.push_back( TestOptions::Texture() );
    opt.textures[1].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.png" ;
    opt.textures[1].bakeToState = false;
    opt.meshes[0].meshResolution = { 2, 2 };
    opt.meshes[0].textures = { 0, 1 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingStateTexture_Mixed" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    EXPECT_GE( ommArray.getNumOmms(), 1u );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingStateTexture, Pitch )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.png" ;
    opt.textures[0].bakeToState = true;
    opt.textures[0].statePaddingInBits = 68;
    opt.meshes[0].meshResolution = { 1, 1 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingStateTexture_Pitch" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    EXPECT_GE( ommArray.getNumOmms(), 1u );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingStateTexture, UclampVwrap )
{
    TestOptions opt = {};
    opt.textures[0].addressMode[0] = cudaAddressModeClamp;
    opt.textures[0].addressMode[1] = cudaAddressModeWrap;
    opt.meshes[0].meshResolution = { 3, 3 };
    opt.meshes[0].uvMin = { -1, -1 };
    opt.meshes[0].uvMax = { 2, 2 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingStateTexture_UclampVwrap" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    EXPECT_GE( ommArray.getNumOmms(), 1u );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingStateTexture, UwrapVclamp )
{
    TestOptions opt = {};
    opt.textures[0].addressMode[0] = cudaAddressModeWrap;
    opt.textures[0].addressMode[1] = cudaAddressModeClamp;
    opt.meshes[0].meshResolution = { 3, 3 };
    opt.meshes[0].uvMin = { -1, -1 };
    opt.meshes[0].uvMax = { 2, 2 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingStateTexture_UwrapVclamp" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    EXPECT_GE( ommArray.getNumOmms(), 1u );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingStateTexture, UclampVclamp )
{
    TestOptions opt = {};
    opt.textures[0].addressMode[0] = cudaAddressModeClamp;
    opt.textures[0].addressMode[1] = cudaAddressModeClamp;
    opt.meshes[0].meshResolution = { 3, 3 };
    opt.meshes[0].uvMin = { -1, -1 };
    opt.meshes[0].uvMax = { 2, 2 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingStateTexture_UclamVclamp" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    EXPECT_GE( ommArray.getNumOmms(), 1u );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingStateTexture, UwrapVwrap )
{
    TestOptions opt = {};
    opt.textures[0].addressMode[0] = cudaAddressModeWrap;
    opt.textures[0].addressMode[1] = cudaAddressModeWrap;
    opt.meshes[0].meshResolution = { 2, 1 };
    
    cuOmmBaking::Result res = buildOmm( opt );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithoutRepeat = ommArray.getNumOmms();
    uint32_t usageWithoutRepeat = ommArray.getNumOmmUses();

    opt.meshes[0].meshResolution = { 10, 5 };
    opt.meshes[0].uvMin = { -2, -2 };
    opt.meshes[0].uvMax = { 3, 3 };
    res = runTest( opt, "OmmBakingStateTexture_UwrapVwrap" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithRepeat = ommArray.getNumOmms();
    uint32_t usageWithRepeat = ommArray.getNumOmmUses();

    EXPECT_GE( numVmsWithoutRepeat, 2u );
    EXPECT_EQ( numVmsWithoutRepeat, numVmsWithRepeat );

    EXPECT_GE( usageWithoutRepeat, 4u );
    EXPECT_EQ( usageWithRepeat, 100u );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingStateTexture, UmirrorVmirror )
{
    TestOptions opt = {};
    opt.textures[0].addressMode[0] = cudaAddressModeMirror;
    opt.textures[0].addressMode[1] = cudaAddressModeMirror;
    opt.meshes[0].meshResolution = { 2, 1 };

    cuOmmBaking::Result res = buildOmm( opt );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithoutRepeat = ommArray.getNumOmms();
    uint32_t usageWithoutRepeat = ommArray.getNumOmmUses();

    opt.meshes[0].meshResolution = { 10, 5 };
    opt.meshes[0].uvMin = { -2, -2 };
    opt.meshes[0].uvMax = { 3, 3 };
    res = runTest( opt, "OmmBakingStateTexture_UmirrorVmirror" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithRepeat = ommArray.getNumOmms();
    uint32_t usageWithRepeat = ommArray.getNumOmmUses();

    EXPECT_GE( numVmsWithoutRepeat, 2u );
    EXPECT_EQ( numVmsWithoutRepeat * 4, numVmsWithRepeat );

    EXPECT_GE( usageWithoutRepeat, 4u );
    EXPECT_EQ( usageWithRepeat, 100u );

    if( HasFatalFailure() )
        return;
}

TEST_F( OmmBakingStateTexture, UmirrorVwrap )
{
    TestOptions opt = {};
    opt.textures[0].addressMode[0] = cudaAddressModeMirror;
    opt.textures[0].addressMode[1] = cudaAddressModeWrap;
    opt.meshes[0].meshResolution = { 2, 1 };

    cuOmmBaking::Result res = buildOmm( opt );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithoutRepeat = ommArray.getNumOmms();
    uint32_t usageWithoutRepeat = ommArray.getNumOmmUses();

    opt.meshes[0].meshResolution = { 10, 5 };
    opt.meshes[0].uvMin = { -2, -2 };
    opt.meshes[0].uvMax = { 3, 3 };
    res = runTest( opt, "OmmBakingStateTexture_UmirrorVwrap" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithRepeat = ommArray.getNumOmms();
    uint32_t usageWithRepeat = ommArray.getNumOmmUses();

    EXPECT_GE( numVmsWithoutRepeat, 2u );
    EXPECT_EQ( numVmsWithoutRepeat * 2, numVmsWithRepeat );

    EXPECT_LE( usageWithoutRepeat, 4u );
    EXPECT_GE( usageWithoutRepeat, numVmsWithoutRepeat );
    EXPECT_LE( usageWithRepeat, 100u );
    EXPECT_GE( usageWithRepeat, numVmsWithRepeat );

    if( HasFatalFailure() )
        return;
}

TEST_F( OmmBakingStateTexture, UwrapVmirror )
{
    TestOptions opt = {};
    opt.textures[0].addressMode[0] = cudaAddressModeWrap;
    opt.textures[0].addressMode[1] = cudaAddressModeMirror;
    opt.meshes[0].meshResolution = { 2, 1 };

    cuOmmBaking::Result res = buildOmm( opt );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithoutRepeat = ommArray.getNumOmms();
    uint32_t usageWithoutRepeat = ommArray.getNumOmmUses();

    opt.meshes[0].meshResolution = { 10, 5 };
    opt.meshes[0].uvMin = { -2, -2 };
    opt.meshes[0].uvMax = { 3, 3 };
    res = runTest( opt, "OmmBakingStateTexture_UwrapVmirror" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithRepeat = ommArray.getNumOmms();
    uint32_t usageWithRepeat = ommArray.getNumOmmUses();

    EXPECT_GE( numVmsWithoutRepeat, 2u );
    EXPECT_EQ( numVmsWithoutRepeat * 2, numVmsWithRepeat );

    EXPECT_LE( usageWithoutRepeat, 4u );
    EXPECT_GE( usageWithoutRepeat, numVmsWithoutRepeat );
    EXPECT_LE( usageWithRepeat, 100u );
    EXPECT_GE( usageWithRepeat, numVmsWithRepeat );

    if( HasFatalFailure() )
        return;
}

class OmmBakingOptionsTest : public OmmBakingTest {};

TEST_F( OmmBakingOptionsTest, Format )
{
    TestOptions opt = {};
    opt.options.format = OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingOptionsTest_Format" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    uint32_t numOmms = ommArray.getNumOmms();

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingOptionsTest, MaximumSize )
{
    TestOptions opt = {};
    opt.options.maximumSizeInBytes = 1;
    opt.meshes[0].meshResolution = { 2, 2 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingOptionsTest_MaximumSize" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    uint32_t numOmms = ommArray.getNumOmms();
    EXPECT_LE( numOmms, 1u );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingOptionsTest, FilterWidth )
{
    TestOptions opt = {};
    opt.filterKernelWidthInTexels = 10.f;
    opt.textures[0].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.png" ;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingOptionsTest_FilterWidth" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingOptionsTest, SubdivisionScale )
{
    TestOptions opt = {};
    opt.options.subdivisionScale = 10.f;
    opt.meshes[0].meshResolution = { 1, 1 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingOptionsTest_SubdivisionScale" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingOptionsTest, PostBakeInfo )
{
    TestOptions opt = {};
    opt.options.maximumSizeInBytes = 1024;
    opt.options.flags = cuOmmBaking::BakeFlags::ENABLE_POST_BAKE_INFO;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingOptionsTest_PostBakeInfo" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    EXPECT_LE( ommArray.getPostBakeInfo().compactedSizeInBytes, opt.options.maximumSizeInBytes );
    EXPECT_GT( ommArray.getPostBakeInfo().compactedSizeInBytes, 0 );

    uint32_t numOmms = ommArray.getNumOmms();
    EXPECT_GE( numOmms, 1u );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

class OmmBakingInputTest : public OmmBakingTest {};

TEST_F( OmmBakingInputTest, MultiInput )
{
    TestOptions opt = {};
    opt.meshes.push_back( TestOptions::MeshOptions() );
    opt.meshes[0].meshResolution = { 1, 1 };
    opt.meshes[1].meshResolution = { 1, 1 };
    opt.meshes[1].uvMin = { 1, 1 }, opt.meshes[1].uvMax = { 2, 2 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingInputTest_MultiInput" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    uint32_t numOmms = ommArray.getNumOmms();
    EXPECT_LE( numOmms, 2u );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingInputTest, Subdivision )
{
    TestOptions opt = {};
    opt.options.maximumSizeInBytes = /*2 triangles*/2 * (/*large input*/4 + /*small input*/1) * ( 1 << ( 2 * /*level*/5 - 2 ) ) + /*precision guard*/1;
    opt.textures[0].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.png" ;
    opt.meshes.push_back( TestOptions::MeshOptions() );
    opt.meshes[0].meshResolution = { 1, 1 };
    opt.meshes[1].meshResolution = { 1, 1 };
    opt.meshes[1].uvMin = { 1, 1 }, opt.meshes[1].uvMax = { 3, 3 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingInputTest_Subdivision" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingInputTest, PackedStride )
{
    TestOptions opt = {};
    opt.meshes[0].format.indicesStrideInBytes = sizeof( uint3 );
    opt.meshes[0].format.texCoordStrideInBytes = sizeof( float2 );
    opt.meshes[0].format.verticesStrideInBytes = sizeof( float3 );

    cuOmmBaking::Result res = runTest( opt, "OmmBakingInputTest_PackedStride" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingInputTest, WideStride )
{
    TestOptions opt = {};
    opt.meshes[0].format.indicesStrideInBytes = 64;
    opt.meshes[0].format.texCoordStrideInBytes = 48;
    opt.meshes[0].format.verticesStrideInBytes = 96;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingInputTest_WideStride" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}


TEST_F( OmmBakingInputTest, IndicesFormatNone )
{
    TestOptions opt = {};
    opt.meshes[0].format.indexFormat = cuOmmBaking::IndexFormat::NONE;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingInputTest_IndicesFormatNone" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingInputTest, IndicesFormatShort3 )
{
    TestOptions opt = {};
    opt.meshes[0].format.indexFormat = cuOmmBaking::IndexFormat::I16_UINT;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingInputTest_IndicesFormatShort3" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingInputTest, PreTransform )
{
    const float mtrx[6] = { 0.f, 2.0f, 0.5f, 3.0f, 0.f, 0.25f };

    TestOptions opt = {};
    opt.meshes[0].transform = mtrx;
    
    cuOmmBaking::Result res = runTest( opt, "OmmBakingInputTest_PreTransform" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingInputTest, TextureIndices )
{
    TestOptions opt = {};
    opt.textures.push_back( TestOptions::Texture() );
    opt.textures[1].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.png" ;
    opt.meshes[0].textures.push_back( 1 );
    opt.meshes[0].meshResolution = { 2,2 };
    opt.meshes[0].uvMax = { 2,2 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingInputTest_TextureIndices" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingInputTest, TextureIndicesFormatShort )
{
    TestOptions opt = {};
    opt.textures.push_back( TestOptions::Texture() );
    opt.textures[1].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.png" ;
    opt.meshes[0].textures.push_back( 1 );
    opt.meshes[0].meshResolution = { 2,2 };
    opt.meshes[0].uvMax = { 2,2 };
    opt.meshes[0].format.textureIndexFormat = cuOmmBaking::IndexFormat::I16_UINT;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingInputTest_TextureIndicesFormatShort" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingInputTest, TextureIndicesWideStride )
{
    TestOptions opt = {};
    opt.textures.push_back( TestOptions::Texture() );
    opt.textures[1].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.png" ;
    opt.meshes[0].textures.push_back( 1 );
    opt.meshes[0].meshResolution = { 4,4 };
    opt.meshes[0].uvMax = { 2,2 };
    opt.meshes[0].format.textureIndicesStrideInBytes = 8;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingInputTest_TextureIndicesWideStride" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingInputTest, MultiTexture )
{
    TestOptions opt = {};

    opt.textures.push_back( TestOptions::Texture() );
    opt.textures[1].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.png" ;

    opt.meshes[0].textures = { 0, 1 };
    opt.meshes[0].meshResolution = { 4,4 };
    opt.meshes[0].uvMax = { 2,2 };

    opt.meshes.push_back( TestOptions::MeshOptions() );
    opt.meshes[1].textures = { 1, 0 };
    opt.meshes[1].meshResolution = { 10, 10 };
    opt.meshes[1].uvMin = { 2, 2 }, opt.meshes[1].uvMax = { 4, 4 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingInputTest_MultiTexture" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingInputTest, ManyTriangles )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.png" ;
    opt.meshes[0].meshResolution = { 256, 256 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingInputTest_ManyTriangles" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

class OmmBakingCudaTextureTest : public OmmBakingTest {};

TEST_F( OmmBakingCudaTextureTest, Linear )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.png" ;
    opt.textures[0].filterMode = cudaFilterModeLinear;
    opt.meshes[0].meshResolution = { 1, 1 };
    
    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_Linear" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, Point )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.png" ;
    opt.textures[0].filterMode = cudaFilterModePoint;
    opt.meshes[0].meshResolution = { 1, 1 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_Point" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, OddSized )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/odd_sized.png" ;
    opt.textures[0].readMode = cudaReadModeElementType;
    opt.textures[0].desc = cudaCreateChannelDesc<unsigned char>();
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 255;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_OddSized" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

#if CUDA_VERSION >= 11050

TEST_F( OmmBakingCudaTextureTest, BC1 )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures/DuckHole/DuckHoleDXT1.dds" ;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_BC1" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, BC3 )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures/DuckHole/DuckHoleDXT5.dds" ;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_BC3" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

#endif // CUDA_VERSION 

TEST_F( OmmBakingCudaTextureTest, ReadModeUChar4 )
{
    TestOptions opt = {};
    opt.textures[0].readMode = cudaReadModeElementType;
    opt.textures[0].desc = cudaCreateChannelDesc<uchar4>();
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 255;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ReadModeUChar4" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ReadModeChar4 )
{
    TestOptions opt = {};
    opt.textures[0].readMode = cudaReadModeElementType;
    opt.textures[0].desc = cudaCreateChannelDesc<char4>();
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 127;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ReadModeChar4" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}
TEST_F( OmmBakingCudaTextureTest, ReadModeUShort4 )
{
    TestOptions opt = {};
    opt.textures[0].readMode = cudaReadModeElementType;
    opt.textures[0].desc = cudaCreateChannelDesc<ushort4>();
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 65535;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ReadModeUShort4" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ReadModeShort4 )
{
    TestOptions opt = {};
    opt.textures[0].readMode = cudaReadModeElementType;
    opt.textures[0].desc = cudaCreateChannelDesc<short4>();
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 32767;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ReadModeShort4" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ReadModeUInt4 )
{
    TestOptions opt = {};
    opt.textures[0].readMode = cudaReadModeElementType;
    opt.textures[0].desc = cudaCreateChannelDesc<uint4>();
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 0xF0000000;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ReadModeUInt4" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ReadModeInt4 )
{
    TestOptions opt = {};
    opt.textures[0].readMode = cudaReadModeElementType;
    opt.textures[0].desc = cudaCreateChannelDesc<int4>();
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 0x7FF00000;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ReadModeInt4" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ReadModeUChar )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_mask.png" ;
    opt.textures[0].readMode = cudaReadModeElementType;
    opt.textures[0].desc = cudaCreateChannelDesc<unsigned char>();
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 255;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ReadModeUChar" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ReadModeChar )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_mask.png" ;
    opt.textures[0].readMode = cudaReadModeElementType;
    opt.textures[0].desc = cudaCreateChannelDesc<char>();
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 127;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ReadModeChar" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}
TEST_F( OmmBakingCudaTextureTest, ReadModeUShort )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_mask.png" ;
    opt.textures[0].readMode = cudaReadModeElementType;
    opt.textures[0].desc = cudaCreateChannelDesc<unsigned short>();
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 65535;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ReadModeUShort" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ReadModeShort )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_mask.png" ;
    opt.textures[0].readMode = cudaReadModeElementType;
    opt.textures[0].desc = cudaCreateChannelDesc<short>();
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 32767;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ReadModeShort" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ReadModeUInt )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_mask.png" ;
    opt.textures[0].readMode = cudaReadModeElementType;
    opt.textures[0].desc = cudaCreateChannelDesc<unsigned int>();
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 0xF0000000;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ReadModeUInt" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ReadModeInt )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_mask.png" ;
    opt.textures[0].readMode = cudaReadModeElementType;
    opt.textures[0].desc = cudaCreateChannelDesc<int>();
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 0x7FF00000;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ReadModeInt" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ChannelX )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_rainbow.png" ;
    opt.textures[0].alphaMode = cuOmmBaking::CudaTextureAlphaMode::CHANNEL_X;
    opt.textures[0].transparencyCutoff = 0.25f;
    opt.textures[0].opacityCutoff = 0.75f;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ChannelX" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ChannelY )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_rainbow.png" ;
    opt.textures[0].alphaMode = cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Y;
    opt.textures[0].transparencyCutoff = 0.25f;
    opt.textures[0].opacityCutoff = 0.75f;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ChannelY" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ChannelZ )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_rainbow.png" ;
    opt.textures[0].alphaMode = cuOmmBaking::CudaTextureAlphaMode::CHANNEL_Z;
    opt.textures[0].transparencyCutoff = 0.25f;
    opt.textures[0].opacityCutoff = 0.75f;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ChannelZ" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ChannelW )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_inverted.png" ;
    opt.textures[0].alphaMode = cuOmmBaking::CudaTextureAlphaMode::CHANNEL_W;
    opt.textures[0].transparencyCutoff = 0.25f;
    opt.textures[0].opacityCutoff = 0.75f;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ChannelW" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, RGBIntensity )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_rainbow.png" ;
    opt.textures[0].alphaMode = cuOmmBaking::CudaTextureAlphaMode::RGB_INTENSITY;
    opt.textures[0].transparencyCutoff = 0.25f;
    opt.textures[0].opacityCutoff = 0.75f;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_RGBIntensity" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, DefaultUChar4 )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_rainbow.png" ;
    opt.textures[0].desc = cudaCreateChannelDesc<uchar4>();
    opt.textures[0].transparencyCutoff = 0.25f;
    opt.textures[0].opacityCutoff = 0.75f;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_DefaultUChar4" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, DefaultUChar2 )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_rainbow.png" ;
    opt.textures[0].desc = cudaCreateChannelDesc<uchar2>();
    opt.textures[0].transparencyCutoff = 0.25f;
    opt.textures[0].opacityCutoff = 0.75f;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_DefaultUChar2" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, DefaultUChar1 )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_rainbow.png" ;
    opt.textures[0].desc = cudaCreateChannelDesc<unsigned char>();
    opt.textures[0].transparencyCutoff = 0.25f;
    opt.textures[0].opacityCutoff = 0.75f;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_DefaultUChar1" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, OpacityCutoff )
{
    TestOptions opt = {};
    opt.textures[0].transparencyCutoff   = 0.f;
    opt.textures[0].opacityCutoff = 2.f;  // force all opaque surfaces to unknown

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_OpacityCutoff" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, OpacityCutoffDefaultNormalizedFloat )
{
    TestOptions opt = {};
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 0.f;  // force all opaque surfaces to unknown

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_OpacityCutoffDefaultNormalizedFloat" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, OpacityCutoffDefaultElementType )
{
    TestOptions opt = {};
    opt.textures[0].transparencyCutoff = 0.f;
    opt.textures[0].opacityCutoff = 0.f;  // force all opaque surfaces to unknown
    opt.textures[0].desc = cudaCreateChannelDesc<uchar4>();
    opt.textures[0].readMode = cudaReadModeElementType;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_OpacityCutoffDefaultElementType" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, AlphaCutoff )
{
    TestOptions opt = {};
    opt.textures[0].transparencyCutoff = -1.f; // force all transparent surfaces to unknown
    opt.textures[0].opacityCutoff = 1.f;

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_AlphaCutoff" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, Layered )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures/3dmd/leaf_layered.dds" ;
    opt.textures[0].transparencyCutoff = 0.025f;
    opt.textures[0].opacityCutoff = 0.975f;

    // build a single vm compatable with all layers
    cuOmmBaking::Result res = buildOmm( opt );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    // render all 4 layers
    for( uint32_t i = 0; i < 4; ++i )
    {
        opt.layer = i;
        res = renderOmm( opt, std::string( "OmmBakingCudaTextureTest_Layer" ) + std::to_string( i ) );
        ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

        if( HasFatalFailure() )
            return;
        // compareImage();
    }
}

TEST_F( OmmBakingCudaTextureTest, MipMap )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures/DuckHole/DuckHole.dds" ;
    opt.meshes[0].meshResolution = { 1, 1 };

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_MipMap" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ChannelFormatUChar )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_mask.png" ;
    opt.textures[0].transparencyCutoff = 0.025f;
    opt.textures[0].opacityCutoff = 0.975f;
    opt.textures[0].desc = cudaCreateChannelDesc<unsigned char>();

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ChannelFormatUChar" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ChannelFormatUShort )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_mask.png" ;
    opt.textures[0].transparencyCutoff = 0.025f;
    opt.textures[0].opacityCutoff = 0.975f;
    opt.textures[0].desc = cudaCreateChannelDesc<unsigned short>();

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ChannelFormatUShort" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ChannelFormatFloat )
{
    TestOptions opt = {};
    opt.textures[0].texture = getSourceDir() + "/Textures//3dmd/leaf_mask.png" ;
    opt.textures[0].transparencyCutoff = 0.025f;
    opt.textures[0].opacityCutoff = 0.975f;
    opt.textures[0].desc = cudaCreateChannelDesc<unsigned short>();

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ChannelFormatFloat" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ChannelFormatUShort4 )
{
    TestOptions opt = {};
    opt.textures[0].transparencyCutoff = 0.025f;
    opt.textures[0].opacityCutoff = 0.975f;
    opt.textures[0].desc = cudaCreateChannelDesc<ushort4>();

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ChannelFormatUShort4" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingCudaTextureTest, ChannelFormatFloat4 )
{
    TestOptions opt = {};
    opt.textures[0].transparencyCutoff = 0.025f;
    opt.textures[0].opacityCutoff = 0.975f;
    opt.textures[0].desc = cudaCreateChannelDesc<float4>();

    cuOmmBaking::Result res = runTest( opt, "OmmBakingCudaTextureTest_ChannelFormatFloat4" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );

    if( HasFatalFailure() )
        return;
    // compareImage();
}

TEST_F( OmmBakingTest, AlignedUwrapVwrap )
{
    TestOptions opt = {};
    opt.textures[0].addressMode[0] = cudaAddressModeWrap;
    opt.textures[0].addressMode[1] = cudaAddressModeWrap;
    opt.meshes[0].meshResolution = { 2, 1 };

    cuOmmBaking::Result res = buildOmm( opt );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithoutRepeat = ommArray.getNumOmms();
    uint32_t usageWithoutRepeat = ommArray.getNumOmmUses();

    opt.meshes[0].meshResolution = { 10, 5 };
    opt.meshes[0].uvMin = { -2, -2 };
    opt.meshes[0].uvMax = { 3, 3 };
    res = runTest( opt, "OmmBakingCudaTextureTest_AlignedUwrapVwrap" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithRepeat = ommArray.getNumOmms();
    uint32_t usageWithRepeat = ommArray.getNumOmmUses();

    EXPECT_GE( numVmsWithoutRepeat, 2u );
    EXPECT_EQ( numVmsWithoutRepeat, numVmsWithRepeat );

    EXPECT_GE( usageWithoutRepeat, 4u );
    EXPECT_EQ( usageWithRepeat, 100u );

    if( HasFatalFailure() )
        return;
}

TEST_F( OmmBakingTest, AlignedUmirrorVmirror )
{
    TestOptions opt = {};
    opt.textures[0].addressMode[0] = cudaAddressModeMirror;
    opt.textures[0].addressMode[1] = cudaAddressModeMirror;
    opt.meshes[0].meshResolution = { 2, 1 };

    cuOmmBaking::Result res = buildOmm( opt );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithoutRepeat = ommArray.getNumOmms();
    uint32_t usageWithoutRepeat  = ommArray.getNumOmmUses();

    opt.meshes[0].meshResolution = { 10, 5 };
    opt.meshes[0].uvMin = { -2, -2 };
    opt.meshes[0].uvMax = { 3, 3 };
    res = runTest( opt, "OmmBakingCudaTextureTest_AlignedUmirrorVmirror" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithRepeat = ommArray.getNumOmms();
    uint32_t usageWithRepeat  = ommArray.getNumOmmUses();

    EXPECT_GE( numVmsWithoutRepeat, 2u );
    EXPECT_EQ( numVmsWithoutRepeat * 4, numVmsWithRepeat );

    EXPECT_GE( usageWithoutRepeat, 4u );
    EXPECT_EQ( usageWithRepeat, 100u );

    if( HasFatalFailure() )
        return;
}

TEST_F( OmmBakingTest, AlignedUwrapVmirror )
{
    TestOptions opt = {};
    opt.textures[0].addressMode[0] = cudaAddressModeWrap;
    opt.textures[0].addressMode[1] = cudaAddressModeMirror;
    opt.meshes[0].meshResolution = { 2, 1 };

    cuOmmBaking::Result res = buildOmm( opt );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithoutRepeat = ommArray.getNumOmms();
    uint32_t usageWithoutRepeat = ommArray.getNumOmmUses();

    opt.meshes[0].meshResolution = { 10, 5 };
    opt.meshes[0].uvMin = { -2, -2 };
    opt.meshes[0].uvMax = { 3, 3 };
    res = runTest( opt, "OmmBakingCudaTextureTest_AlignedUwrapVmirror" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithRepeat = ommArray.getNumOmms();
    uint32_t usageWithRepeat = ommArray.getNumOmmUses();

    EXPECT_GE( numVmsWithoutRepeat, 2u );
    EXPECT_EQ( numVmsWithoutRepeat * 2, numVmsWithRepeat );

    EXPECT_LE( usageWithoutRepeat, 4u );
    EXPECT_GE( usageWithoutRepeat, numVmsWithoutRepeat );
    EXPECT_LE( usageWithRepeat, 100u );
    EXPECT_GE( usageWithRepeat, numVmsWithRepeat );

    if( HasFatalFailure() )
        return;
}

TEST_F( OmmBakingTest, AlignedUmirrorVwrap )
{
    TestOptions opt = {};
    opt.textures[0].addressMode[0] = cudaAddressModeMirror;
    opt.textures[0].addressMode[1] = cudaAddressModeWrap;
    opt.meshes[0].meshResolution = { 2, 1 };

    cuOmmBaking::Result res = buildOmm( opt );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithoutRepeat = ommArray.getNumOmms();
    uint32_t usageWithoutRepeat = ommArray.getNumOmmUses();

    opt.meshes[0].meshResolution = { 10, 5 };
    opt.meshes[0].uvMin = { -2, -2 };
    opt.meshes[0].uvMax = { 3, 3 };
    res = runTest( opt, "OmmBakingCudaTextureTest_AlignedUmirrorVwrap" );
    ASSERT_EQ( res, cuOmmBaking::Result::SUCCESS );
    uint32_t numVmsWithRepeat = ommArray.getNumOmms();
    uint32_t usageWithRepeat = ommArray.getNumOmmUses();

    EXPECT_GE( numVmsWithoutRepeat, 2u );
    EXPECT_EQ( numVmsWithoutRepeat * 2, numVmsWithRepeat );

    EXPECT_LE( usageWithoutRepeat, 4u );
    EXPECT_GE( usageWithoutRepeat, numVmsWithoutRepeat );
    EXPECT_LE( usageWithRepeat, 100u );
    EXPECT_GE( usageWithRepeat, numVmsWithRepeat );

    if( HasFatalFailure() )
        return;
}
