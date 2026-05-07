// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandMaterial/MaterialLoader.h>

#include <gmock/gmock.h>

namespace otk {
namespace testing {

class MockMaterialLoader : public ::testing::StrictMock<demandMaterial::MaterialLoader>
{
  public:
    ~MockMaterialLoader() override = default;

    MOCK_METHOD( const char*, getCHFunctionName, (), ( const, override ) );
    MOCK_METHOD( demandMaterial::uint_t, add, (), ( override ) );
    MOCK_METHOD( void, remove, ( demandMaterial::uint_t ), ( override ) );
    MOCK_METHOD( std::vector<demandMaterial::uint_t>, requestedMaterialIds, (), ( const, override ) );
    MOCK_METHOD( void, clearRequestedMaterialIds, (), ( override ) );
    MOCK_METHOD( bool, getRecycleProxyIds, (), ( const, override ) );
    MOCK_METHOD( void, setRecycleProxyIds, (bool), ( override ) );
};

}  // namespace testing
}  // namespace otk
