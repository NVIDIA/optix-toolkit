// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>

#include <gmock/gmock.h>

#include <memory>

namespace demandPbrtScene {
namespace testing {

class MockMaterialLoader : public ::testing::StrictMock<demandMaterial::MaterialLoader>
{
  public:
    ~MockMaterialLoader() override = default;

    MOCK_METHOD( const char*, getCHFunctionName, (), ( const, override ) );
    MOCK_METHOD( uint_t, add, (), ( override ) );
    MOCK_METHOD( void, remove, ( uint_t ), ( override ) );
    MOCK_METHOD( std::vector<uint_t>, requestedMaterialIds, (), ( const, override ) );
    MOCK_METHOD( void, clearRequestedMaterialIds, (), ( override ) );
    MOCK_METHOD( bool, getRecycleProxyIds, (), ( const, override ) );
    MOCK_METHOD( void, setRecycleProxyIds, (bool), ( override ) );
};

using MockMaterialLoaderPtr = std::shared_ptr<MockMaterialLoader>;

inline MockMaterialLoaderPtr createMockMaterialLoader()
{
    return std::make_shared<MockMaterialLoader>();
}

}  // namespace testing
}  // namespace demandPbrtScene
