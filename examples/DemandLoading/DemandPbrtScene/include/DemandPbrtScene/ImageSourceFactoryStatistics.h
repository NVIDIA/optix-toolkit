// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ImageSource/ImageSourceCacheStatistics.h>

namespace demandPbrtScene {

struct ImageSourceFactoryStatistics
{
    imageSource::CacheStatistics fileSources;
    imageSource::CacheStatistics alphaSources;
    imageSource::CacheStatistics diffuseSources;
    imageSource::CacheStatistics skyboxSources;
};

}  // namespace demandPbrtScene
