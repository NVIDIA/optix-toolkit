//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
