// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/DemandGeometry/GeometryLoader.h>

#include <optix.h>

#include <cuda.h>

#include <gmock/gmock.h>

#include <iostream>
#include <memory>

namespace demandGeometry {

inline std::ostream& operator<<( std::ostream& str, const Context& val )
{
    return str << "DGContext{" << val.proxies << "}";
}

inline bool operator==( const Context& lhs, const Context& rhs )
{
    return lhs.proxies == rhs.proxies;
}

inline bool operator!=( const Context& lhs, const Context& rhs )
{
    return !( lhs == rhs );
}

}  // namespace demandGeometry

namespace otk {
namespace testing {

class MockGeometryLoader : public ::testing::StrictMock<::demandGeometry::GeometryLoader>
{
  public:
    ~MockGeometryLoader() override = default;

    MOCK_METHOD( ::demandGeometry::uint_t, add, (const OptixAabb&), ( override ) );
    MOCK_METHOD( void, remove, ( ::demandGeometry::uint_t ), ( override ) );
    MOCK_METHOD( void, copyToDevice, (), ( override ) );
    MOCK_METHOD( void, copyToDeviceAsync, ( CUstream ), ( override ) );
    MOCK_METHOD( std::vector<::demandGeometry::uint_t>, requestedProxyIds, (), ( const, override ) );
    MOCK_METHOD( void, clearRequestedProxyIds, (), ( override ) );
    MOCK_METHOD( void, setSbtIndex, ( ::demandGeometry::uint_t ), ( override ) );
    MOCK_METHOD( OptixTraversableHandle, createTraversable, ( OptixDeviceContext, CUstream ), ( override ) );
    MOCK_METHOD( demandGeometry::Context, getContext, (), ( const, override ) );
    MOCK_METHOD( const char*, getCHFunctionName, (), ( const, override ) );
    MOCK_METHOD( const char*, getISFunctionName, (), ( const, override ) );
    MOCK_METHOD( int, getNumAttributes, (), ( const, override ) );
    MOCK_METHOD( bool, getRecycleProxyIds, (), ( const, override ) );
    MOCK_METHOD( void, setRecycleProxyIds, (bool), ( override ) );
};

using MockGeometryLoaderPtr = std::shared_ptr<MockGeometryLoader>;

inline MockGeometryLoaderPtr createMockGeometryLoader()
{
    return std::make_shared<MockGeometryLoader>();
}

}  // namespace testing
}  // namespace otk
