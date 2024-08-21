// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <MaterialResolver.h>

#include <gmock/gmock.h>

#include <memory>

namespace demandPbrtScene {
namespace testing {

class MockMaterialResolver : public ::testing::StrictMock<MaterialResolver>
{
  public:
    ~MockMaterialResolver() override = default;

    MOCK_METHOD( MaterialResolverStats, getStatistics, (), ( const, override ) );
    MOCK_METHOD( bool, resolveMaterialForGeometry, (uint_t, const GeometryInstance&, SceneSyncState&), ( override ) );
    MOCK_METHOD( void, resolveOneMaterial, (), ( override ) );
    MOCK_METHOD( MaterialResolution, resolveRequestedProxyMaterials, (CUstream, const FrameStopwatch&, SceneSyncState&), ( override ) );
};

using MockMaterialResolverPtr = std::shared_ptr<MockMaterialResolver>;

inline MockMaterialResolverPtr createMockMaterialResolver()
{
    return std::make_shared<MockMaterialResolver>();
}

}  // namespace testing
}  // namespace demandPbrtScene
