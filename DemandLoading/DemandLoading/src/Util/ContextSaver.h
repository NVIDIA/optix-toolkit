// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <cuda.h>

namespace demandLoading {

/// Save and restore CUDA context.
class ContextSaver
{
  public:
    ContextSaver() { OTK_ERROR_CHECK( cuCtxGetCurrent( &m_context ) ); }

    ~ContextSaver() { OTK_ERROR_CHECK_NOTHROW( cuCtxSetCurrent( m_context ) ); }

  private:
    CUcontext m_context;
};

} // namespace demandLoading

