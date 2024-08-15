// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Util/ImageBuffer.h>

#include <cuda_runtime.h>
#include <vector_types.h>

#include <cstdlib>
#include <chrono>
#include <vector>

struct GLFWwindow;

namespace otk {

void displayBufferWindow( const char* title, const ImageBuffer& buffer );

void        initGL();
void        initGLFW();
GLFWwindow* initGLFW( const char* window_title, int width, int height );

// Parse the string of the form <width>x<height> and return numeric values.
void parseDimensions(
        const char* arg,                    // String of form <width>x<height>
        int& width,                         // [out] width
        int& height );                      // [in]  height


// Get current time in seconds for benchmarking/timing purposes.
double currentTime();

// Ensures that width and height have the minimum size to prevent launch errors.
void ensureMinimumSize(
    int& width,                             // Will be assigned the minimum suitable width if too small.
    int& height);                           // Will be assigned the minimum suitable height if too small.

// Ensures that width and height have the minimum size to prevent launch errors.
void ensureMinimumSize(
    unsigned& width,                        // Will be assigned the minimum suitable width if too small.
    unsigned& height);                      // Will be assigned the minimum suitable height if too small.

} // end namespace otk

