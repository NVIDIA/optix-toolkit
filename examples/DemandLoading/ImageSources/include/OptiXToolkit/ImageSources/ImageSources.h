// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ImageSource/ImageSource.h>

#include <memory>
#include <string>

namespace imageSources {

std::shared_ptr<imageSource::ImageSource> createImageSource( const std::string& filename, const std::string& directory = "" );

}  // namespace imageSource
