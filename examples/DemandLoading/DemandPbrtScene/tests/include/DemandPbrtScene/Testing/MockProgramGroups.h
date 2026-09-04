// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <DemandPbrtScene/ProgramGroups.h>
#include <DemandPbrtScene/SceneProxy.h>

#include <gmock/gmock.h>

#include <memory>

namespace demandPbrtScene {
namespace testing {

class MockProgramGroups : public ::testing::StrictMock<ProgramGroups>
{
  public:
    ~MockProgramGroups() override = default;

    MOCK_METHOD( void, cleanup, (), ( override ) );
#ifdef OTK_USE_MDL
    MOCK_METHOD( uint_t, getFallbackMaterialSbtOffset, (const GeometryInstance&), ( override ) );
    MOCK_METHOD( uint_t, getFourierMaterialSbtOffset, (const GeometryInstance&), ( override ) );
    MOCK_METHOD( uint_t, getMdlMaterialSbtOffset, (const GeometryInstance&), ( override ) );
    MOCK_METHOD( MdlMaterialShader, realizeMdlMaterialShader, ( const GeometryInstance&, uint_t ), ( override ) );
    MOCK_METHOD( FourierMaterialResource, realizeFourierMaterialResource, (const GeometryInstance&, const FourierBsdfTable&), ( override ) );

    uint_t reserveMdlMaterialSbtOffset( const GeometryInstance& instance, uint_t ) override
    {
        return getMdlMaterialSbtOffset( instance );
    }

    uint_t realizeMdlMaterialSbtOffset( const GeometryInstance& instance, uint_t ) override
    {
        return instance.instance.sbtOffset;
    }
#endif
    MOCK_METHOD( uint_t, getRealizedMaterialSbtOffset, (const GeometryInstance&), ( override ) );
    MOCK_METHOD( void, initialize, (), ( override ) );
};

using MockProgramGroupsPtr = std::shared_ptr<MockProgramGroups>;

inline MockProgramGroupsPtr createMockProgramGroups()
{
    return std::make_shared<MockProgramGroups>();
}

}  // namespace testing
}  // namespace demandPbrtScene
