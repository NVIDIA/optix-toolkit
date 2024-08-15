// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Memory/MemoryBlockDesc.h>

namespace demandLoading {

struct TransferBufferDesc
{
    CUmemorytype memoryType;
    otk::MemoryBlockDesc memoryBlock;
};

} // namespace demandLoading
