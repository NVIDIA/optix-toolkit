// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
