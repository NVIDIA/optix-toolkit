// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda.h>

#include <cstddef>
#include <string>

namespace demandLoading {

/// Maximum total size (in bytes) of a mipmapped CUDA array whose mip levels can all be sampled
/// correctly on the device.
///
/// CUDA addresses texels within a mipmapped array using a 32-bit byte offset, so any texel whose
/// offset reaches 2^32 (4 GiB) samples as zero (black) on the device -- even though the array
/// creates successfully and CPU readback works.  Because mip level 0 sits at offset 0 and coarser
/// levels are packed at increasing offsets, the coarsest levels are the first to fail as the array
/// grows.  This limit is the same for dense and sparse (CUDA_ARRAY3D_SPARSE) arrays.
const size_t MAX_MIPMAPPED_ARRAY_BYTES = size_t{ 1 } << 32;

/// Total packed size, in bytes, of a mipmapped array with the given descriptor and number of mip
/// levels.  This is the sum over all levels of (bytes per pixel * level width * level height),
/// computed in 64-bit arithmetic.  Bytes per pixel matches imageSource::getBitsPerPixel(), including
/// its promotion of 3 channels to 4.
size_t getMipmappedArrayPackedBytes( const CUDA_ARRAY3D_DESCRIPTOR& desc, unsigned int numMipLevels );

/// Returns true if a mipmapped array with the given descriptor and number of mip levels can be
/// sampled correctly on the device, i.e. its packed size does not exceed MAX_MIPMAPPED_ARRAY_BYTES.
/// When the size is not supported and reason is non-null, a human-readable explanation is stored in
/// *reason.
bool isMipmappedArraySizeSupported( const CUDA_ARRAY3D_DESCRIPTOR& desc, unsigned int numMipLevels, std::string* reason = nullptr );

/// Wrapper around cuMipmappedArrayCreate that guards against the 4 GiB device-sampling limit
/// described above.  If the requested size is not supported, it throws (via OTK_ERROR_CHECK_MSG)
/// with a descriptive message instead of silently creating an array whose coarse mip levels would
/// sample as zero.  Otherwise it forwards to cuMipmappedArrayCreate and throws on any CUDA error.
void createMipmappedArray( CUmipmappedArray* array, const CUDA_ARRAY3D_DESCRIPTOR* desc, unsigned int numMipLevels );

}  // namespace demandLoading
