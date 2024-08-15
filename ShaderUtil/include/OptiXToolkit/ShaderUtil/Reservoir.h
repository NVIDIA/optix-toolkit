// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

template <typename TYPE> struct Reservoir
{
    TYPE y;             // Chosen sample. y = f/phat, where phat is desired distribution
    float wsum = 0.0f;  // Sum of weights. weight = phat/p 
    int m = 0;          // Num samples seen so far

    __forceinline__ __device__ void update( TYPE& x, float w, float rnd, int nsamples = 1 )
    {
        m += nsamples;
        wsum += w;
        if( rnd * wsum < w )
            y = x;
    }

    __forceinline__ __device__ void update( Reservoir& r, float rnd )
    {
        update( r.y, r.wsum, rnd, r.m );
    }
};