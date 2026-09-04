// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace demandPbrtScene {

enum class FourierBsdfTableLoadStatus
{
    SUCCESS,
    FILE_NOT_FOUND,
    INVALID_HEADER,
    TRUNCATED,
    UNSUPPORTED,
    MALFORMED,
};

struct FourierBsdfTable
{
    int         flags{};
    int         nMu{};
    int         nCoefficients{};
    int         maxOrder{};
    int         nChannels{};
    int         nBases{};
    float       eta{};
    std::size_t trailingByteCount{};

    std::vector<float> mu;
    std::vector<float> cdf;
    std::vector<int>   coefficientOffsets;
    std::vector<int>   coefficientCounts;
    std::vector<float> zeroOrderCoefficients;
    std::vector<float> coefficients;
};

struct FourierBsdfTableLoadResult
{
    FourierBsdfTableLoadStatus status{ FourierBsdfTableLoadStatus::FILE_NOT_FOUND };
    FourierBsdfTable           table{};
    std::string                diagnostic;

    explicit operator bool() const { return status == FourierBsdfTableLoadStatus::SUCCESS; }
};

FourierBsdfTableLoadResult loadFourierBsdfTable( const std::string& fileName );

}  // namespace demandPbrtScene
