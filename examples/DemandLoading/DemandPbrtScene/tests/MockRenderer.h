// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <Renderer.h>

#include <gmock/gmock.h>

#include <memory>

namespace demandPbrtScene {
namespace testing {

class MockRenderer : public ::testing::StrictMock<Renderer>
{
  public:
    ~MockRenderer() override = default;

    MOCK_METHOD( void, initialize, ( CUstream ), ( override ) );
    MOCK_METHOD( void, cleanup, (), ( override ) );
    MOCK_METHOD( const otk::DebugLocation&, getDebugLocation, (), ( const override ) );
    MOCK_METHOD( const LookAtParams&, getLookAt, (), ( const override ) );
    MOCK_METHOD( const PerspectiveCamera&, getCamera, (), ( const override ) );
    MOCK_METHOD( Params&, getParams, (), ( override ) );
    MOCK_METHOD( OptixDeviceContext, getDeviceContext, (), ( const, override ) );
    MOCK_METHOD( const OptixPipelineCompileOptions&, getPipelineCompileOptions, (), ( const, override ) );
    MOCK_METHOD( void, setDebugLocation, (const otk::DebugLocation&), ( override ) );
    MOCK_METHOD( void, setCamera, (const PerspectiveCamera&), ( override ) );
    MOCK_METHOD( void, setLookAt, (const LookAtParams&), ( override ) );
    MOCK_METHOD( void, setProgramGroups, (const std::vector<OptixProgramGroup>&), ( override ) );
    MOCK_METHOD( void, beforeLaunch, ( CUstream ), ( override ) );
    MOCK_METHOD( void, launch, (CUstream, uchar4*), ( override ) );
    MOCK_METHOD( void, afterLaunch, (), ( override ) );
    MOCK_METHOD( void, fireOneDebugDump, (), ( override ) );
    MOCK_METHOD( void, setClearAccumulator, (), ( override ) );
};

using MockRendererPtr = std::shared_ptr<MockRenderer>;

inline MockRendererPtr createMockRenderer()
{
    return std::make_shared<MockRenderer>();
}

}  // namespace testing
}  // namespace demandPbrtScene
