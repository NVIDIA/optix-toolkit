//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

