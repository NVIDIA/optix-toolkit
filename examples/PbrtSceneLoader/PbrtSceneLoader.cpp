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

#include <OptiXToolkit/PbrtSceneLoader/SceneLoader.h>

#include "PbrtApiImpl.h"

#include <utility>

namespace otk {
namespace pbrt {

class PbrtSceneLoader : public SceneLoader
{
  public:
    PbrtSceneLoader( const char* programName, std::shared_ptr<Logger> logger, std::shared_ptr<MeshInfoReader> infoReader )
        : m_api( std::make_shared<PbrtApiImpl>( programName, std::move( logger ), std::move( infoReader ) ) )
    {
    }
    ~PbrtSceneLoader() override = default;

    SceneDescriptionPtr parseFile( const std::string& filename ) override { return m_api->parseFile( filename ); }
    SceneDescriptionPtr parseString( const std::string& str ) override { return m_api->parseString( str ); }

  private:
    std::shared_ptr<PbrtApiImpl> m_api;
};

std::shared_ptr<SceneLoader> createSceneLoader( const char*                            programName,
                                                const std::shared_ptr<Logger>&         logger,
                                                const std::shared_ptr<MeshInfoReader>& infoReader )
{
    return std::make_shared<PbrtSceneLoader>( programName, logger, infoReader );
}

}  // namespace pbrt
}  // namespace otk
