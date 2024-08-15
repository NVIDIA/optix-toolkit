// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>

namespace demandPbrtScene {

class CudaContext
{
  public:
    CudaContext( unsigned int deviceIndex );
    ~CudaContext() = default;

    void setCurrent();

    CUstream getStream() const { return m_stream; }

private:
    unsigned int m_deviceIndex{};
    CUcontext    m_cudaContext{};
    CUstream     m_stream{};
};

}  // namespace demandPbrtScene
