// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

// gtest has to come before pbrt stuff
#include <gmock/gmock.h>

#include <OptiXToolkit/PbrtSceneLoader/MeshReader.h>

#include <memory>

namespace demandPbrtScene {
namespace testing {

class MockMeshLoader : public ::testing::StrictMock<otk::pbrt::MeshLoader>
{
  public:
    ~MockMeshLoader() override = default;

    MOCK_METHOD( otk::pbrt::MeshInfo, getMeshInfo, (), ( const, override ) );
    MOCK_METHOD( void, load, (otk::pbrt::MeshData&), ( override ) );
};

using MockMeshLoaderPtr = std::shared_ptr<MockMeshLoader>;

inline MockMeshLoaderPtr createMockMeshLoader()
{
    return std::make_shared<MockMeshLoader>();
}

}  // namespace testing
}  // namespace demandPbrtScene
