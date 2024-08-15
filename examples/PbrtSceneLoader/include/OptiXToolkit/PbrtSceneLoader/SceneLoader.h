// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <memory>
#include <string>

namespace otk {
namespace pbrt {

struct SceneDescription;
using SceneDescriptionPtr = std::shared_ptr<SceneDescription>;

class SceneLoader
{
public:
    virtual ~SceneLoader() = default;

    virtual SceneDescriptionPtr parseFile( const std::string& filename ) = 0;
    virtual SceneDescriptionPtr parseString( const std::string& str )    = 0;
};

class Logger;
class MeshInfoReader;

std::shared_ptr<SceneLoader> createSceneLoader( const char*                            programName,
                                                const std::shared_ptr<Logger>&         logger,
                                                const std::shared_ptr<MeshInfoReader>& infoReader );

}  // namespace pbrt
}  // namespace otk
