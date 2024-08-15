// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cstdlib>

namespace otk {

enum class BufferImageFormat
{
    UNSIGNED_BYTE4,
    FLOAT4,
    FLOAT3
};

struct ImageBuffer
{
    void* data =      nullptr;
    unsigned int      width = 0;
    unsigned int      height = 0;
    BufferImageFormat pixel_format;
};

size_t pixelFormatSize( BufferImageFormat format );

// Floating point image buffers (see BufferImageFormat above) are assumed to be
// linear and will be converted to sRGB when writing to a file format with 8
// bits per channel.  This can be skipped if disable_srgb is set to true.
// Image buffers with format UNSIGNED_BYTE4 are assumed to be in sRGB already
// and will be written like that.
void saveImage( const char* fname, const ImageBuffer& image, bool disableSRGBConversion );

} // namespace otk
