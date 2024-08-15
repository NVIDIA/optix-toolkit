// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define OTK_DEVICE __device__
#    define OTK_HOSTDEVICE __host__ __device__
#    define OTK_INLINE __forceinline__
#    define CONST_STATIC_INIT( ... )
#else
#    define OTK_DEVICE
#    define OTK_HOSTDEVICE
#    define OTK_INLINE inline
#    define CONST_STATIC_INIT( ... ) = __VA_ARGS__
#endif



