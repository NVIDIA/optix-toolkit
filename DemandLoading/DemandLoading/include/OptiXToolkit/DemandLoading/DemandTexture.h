// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file DemandTexture.h
/// Opaque handle for demand-loaded sparse texture.

#include <cuda.h>
#include <vector_types.h>

#include <vector>

namespace demandLoading {

/// Demand-loaded textures are created and owned by the DemandLoader.
/// The methods may be called from multiple threads; the implementation must be threadsafe.
class DemandTexture
{
  public:
    /// Default destructor.
    virtual ~DemandTexture() = default;

    /// Get the texture id, which is used as an index into the device-side sampler array.
    virtual unsigned int getId() const = 0;
};

}  // namespace demandLoading
