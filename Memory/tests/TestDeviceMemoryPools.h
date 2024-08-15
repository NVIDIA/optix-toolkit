// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Memory/DeviceFixedPool.h>
#include <OptiXToolkit/Memory/DeviceRingBuffer.h>

#include <cuda.h>

void launchDeviceRingBufferTest( const otk::DeviceRingBuffer& ringBuffer, char** output, int width );
void launchDeviceFixedPoolTest( const otk::DeviceFixedPool& fixedPool, char** output, int width );
void launchDeviceFixedPoolInterleavedTest( const otk::DeviceFixedPool& fixedPool, char** output, int width );
