//
// Copyright (c) 2023 NVIDIA Corporation.  All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software, related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is strictly
// prohibited.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
// AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
// INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
// SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
// LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
// BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES
//

#include <OptiXToolkit/PbrtSceneLoader/PbrtSceneLoader.h>

#include <OptiXToolkit/PbrtApi/PbrtApi.h>
#include <OptiXToolkit/PbrtSceneLoader/Logger.h>

#include <core/api.h>

namespace otk {
namespace pbrt {

PbrtSceneLoader::PbrtSceneLoader( const char* programName, std::shared_ptr<Logger> logger, std::shared_ptr<Api> api )
    : m_logger( std::move( logger ) )
    , m_api( std::move( api ) )
{
    m_logger->start( programName );
    setApi( m_api.get() );
}

PbrtSceneLoader::~PbrtSceneLoader()
{
    m_logger->stop();
    setApi( nullptr );
}

void PbrtSceneLoader::loadFile( std::string filename )
{
    m_api->parseFile( std::move( filename ) );
}

void PbrtSceneLoader::loadString( std::string text )
{
    m_api->parseString( std::move( text ) );
}

}  // namespace pbrt
}  // namespace otk
