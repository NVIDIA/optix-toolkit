// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

namespace demandPbrtScene {

enum class FourierGpuEvaluationPath
{
    MDL_MEASURED_BSDF,
    PBRT_FOURIER_CALLABLE,
};

struct FourierMdlMeasuredBsdfCapability
{
    bool                     acceptsPbrtBsdfTables{};
    bool                     exposesSampleEvaluatePdfCallables{};
    FourierGpuEvaluationPath selectedPath{ FourierGpuEvaluationPath::PBRT_FOURIER_CALLABLE };
    const char*              reason{};
};

constexpr FourierMdlMeasuredBsdfCapability fourierMdlMeasuredBsdfCapability()
{
    return FourierMdlMeasuredBsdfCapability{
        false,
        false,
        FourierGpuEvaluationPath::PBRT_FOURIER_CALLABLE,
        "MDL measured BSDF resources accept .mbsdf data; PBRT Fourier .bsdf tables require a PBRT-specific GPU "
        "callable until a lossless conversion is available.",
    };
}

}  // namespace demandPbrtScene
