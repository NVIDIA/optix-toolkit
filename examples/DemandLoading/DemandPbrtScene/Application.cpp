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

#include "Application.h"

#include <ProgramGroups.h>

#include "GeometryCache.h"
#include "ImageSourceFactory.h"
#include "Renderer.h"
#include "Scene.h"
#include "SceneProxy.h"
#include "Statistics.h"
#include "UserInterface.h"
#include "UserInterfaceStatistics.h"

#include <OptiXToolkit/DemandGeometry/ProxyInstances.h>
#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Gui/BufferMapper.h>
#include <OptiXToolkit/PbrtSceneLoader/GoogleLogger.h>
#include <OptiXToolkit/PbrtSceneLoader/PlyReader.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneLoader.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <stdexcept>

#include "DemandTextureCache.h"

namespace demandPbrtScene {

using OutputBuffer = otk::CUDAOutputBuffer<uchar4>;

static demandLoading::Options getDemandLoaderOptions()
{
    demandLoading::Options options{};
    options.numPageTableEntries = 6 * 1024 * 1024;
    return options;
}

Application::Application( int argc, char* argv[] )
    : m_options( parseOptions( argc, argv ) )
    , m_cuda( getCudaDeviceIndex() )
    , m_logger( std::make_shared<otk::pbrt::GoogleLogger>( m_options.verboseLoading ? /*info=*/0 : /*warning=*/1 ) )
    , m_infoReader( std::make_shared<ply::InfoReader>() )
    , m_pbrt( createSceneLoader( m_options.program.c_str(), m_logger, m_infoReader ) )
    , m_demandLoader( createDemandLoader( getDemandLoaderOptions() ), demandLoading::destroyDemandLoader )
    , m_geometryLoader( std::make_shared<demandGeometry::ProxyInstances>( m_demandLoader.get() ) )
    , m_materialLoader( demandMaterial::createMaterialLoader( m_demandLoader.get() ) )
    , m_geometryCache( createGeometryCache( createFileSystemInfo() ) )
    , m_imageSourceFactory( createImageSourceFactory( m_options ) )
    , m_proxyFactory( createProxyFactory( m_options, m_geometryCache ) )
    , m_renderer( createRenderer( m_options, m_geometryLoader->getNumAttributes() ) )
    , m_demandTextureCache( createDemandTextureCache( m_demandLoader, m_imageSourceFactory ) )
    , m_programGroups( createProgramGroups( m_geometryLoader, m_materialLoader, m_renderer ) )
    , m_scene( createPbrtScene( m_options, m_pbrt, m_demandTextureCache, m_proxyFactory, m_demandLoader, m_geometryLoader, m_materialLoader, m_programGroups, m_renderer ) )
{
}

void Application::initialize()
{
    Timer          timer;
    const CUstream stream = m_cuda.getStream();
    m_renderer->initialize( stream );
    m_scene->initialize( stream );
    m_statistics.initTime = timer.getSeconds();
}

unsigned int Application::getCudaDeviceIndex()
{
    OTK_ERROR_CHECK( cudaFree( nullptr ) );
    const unsigned int device = demandLoading::getFirstSparseTextureDevice();
    if( device == demandLoading::MAX_DEVICES )
        throw std::runtime_error( "No devices support demand loading" );
    return device;
}

void Application::run()
{
    if( m_options.outFile.empty() )
        runInteractive();
    else
        runToFile();
}

void Application::launch( otk::CUDAOutputBuffer<uchar4>& outputBuffer )
{
    m_cuda.setCurrent();
    Params&  params = m_renderer->getParams();
    CUstream stream = m_cuda.getStream();
    m_sceneUpdated  = m_scene->beforeLaunch( stream, params );
    m_renderer->beforeLaunch( stream );

    {
        otk::BufferMapper<uchar4> mappedBuffer( outputBuffer );
        try
        {
            m_renderer->launch( stream, mappedBuffer );
        }
        catch( const std::runtime_error& e )
        {
            std::cerr << "Error: " << e.what() << std::endl;
            abort();
        }
    }
    m_scene->afterLaunch( stream, params );
    m_renderer->afterLaunch();
}

void Application::updateStats( const UserInterfacePtr& ui )
{
    ++m_stats.numFrames;
    UserInterfaceStatistics stats;
    stats.numFramesRendered  = m_stats.numFrames;
    stats.geometryCache      = m_geometryCache->getStatistics();
    stats.imageSourceFactory = m_imageSourceFactory->getStatistics();
    stats.proxyFactory       = m_proxyFactory->getStatistics();
    stats.scene              = m_scene->getStatistics();
    ui->setStatistics( stats );
}

void Application::runInteractive()
{
    UserInterfacePtr ui{ createUserInterface( m_options, m_renderer, m_scene ) };
    ui->initialize( m_renderer->getLookAt(), m_renderer->getCamera() );
    {
        // OutputBuffer needs to be destroyed before calling UserInterface::cleanup.
        OutputBuffer output{ otk::CUDAOutputBufferType::GL_INTEROP, m_options.width, m_options.height };
        output.setStream( m_cuda.getStream() );

        do
        {
            const bool uiUpdated = ui->beforeLaunch( output );
            // Keep launching for the accumulator
            //if( uiUpdated || m_sceneUpdated )
            {
                launch( output );
            }
            updateStats( ui );
            ui->afterLaunch( output );
        } while( !ui->shouldClose() );
    }

    ui->cleanup();
}

void Application::saveResults( otk::CUDAOutputBuffer<uchar4>& outputBuffer )
{
    otk::ImageBuffer buffer;
    buffer.data         = outputBuffer.getHostPointer();
    buffer.width        = m_options.width;
    buffer.height       = m_options.height;
    buffer.pixel_format = otk::BufferImageFormat::UNSIGNED_BYTE4;
    saveImage( m_options.outFile.c_str(), buffer, false );
}

void Application::runToFile()
{
    if( m_options.debug )
    {
        otk::DebugLocation debug;
        debug.enabled       = true;
        debug.debugIndexSet = true;
        debug.debugIndex    = make_uint3( m_options.debugPixel.x, m_options.debugPixel.y, 0 );
        m_renderer->setDebugLocation( debug );
    }

    otk::CUDAOutputBuffer<uchar4> output( otk::CUDAOutputBufferType::CUDA_DEVICE, m_options.width, m_options.height );
    output.setStream( m_cuda.getStream() );

    for( int i = 0; i < m_options.warmupFrames; ++i )
        launch( output );

    launch( output );
    saveResults( output );
}

void Application::cleanup()
{
    m_scene->cleanup();
    m_renderer->cleanup();
    m_statistics.report();
}

}  // namespace demandPbrtScene
