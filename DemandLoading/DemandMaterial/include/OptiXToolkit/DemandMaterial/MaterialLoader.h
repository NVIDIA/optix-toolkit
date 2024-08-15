// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <memory>
#include <vector>

namespace demandLoading {
class DemandLoader;
}  // namespace demandLoading

namespace demandMaterial {

using uint_t = unsigned int;

class MaterialLoader
{
  public:
    virtual ~MaterialLoader() = default;

    virtual const char* getCHFunctionName() const = 0;

    virtual uint_t add()               = 0;
    virtual void   remove( uint_t id ) = 0;

    virtual std::vector<uint_t> requestedMaterialIds() const = 0;
    virtual void                clearRequestedMaterialIds()  = 0;

    /// Return whether or not proxy ids are recycled as they are removed.
    virtual bool getRecycleProxyIds() const = 0;

    /// Enable or disable whether or not proxy ids are recycled as they are removed.
    /// The default is to not recycle proxy ids.
    virtual void setRecycleProxyIds( bool enable ) = 0;
};

std::shared_ptr<MaterialLoader> createMaterialLoader( demandLoading::DemandLoader* demandLoader );

}  // namespace demandMaterial
