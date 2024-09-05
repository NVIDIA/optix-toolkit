// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSources/DeviceConstantImage.h>
#include <OptiXToolkit/ImageSources/DeviceMandelbrotImage.h>
#include <OptiXToolkit/ImageSources/ImageSources.h>
#include <OptiXToolkit/ImageSources/MultiCheckerImage.h>

#include <memory>

namespace imageSources {

std::shared_ptr<imageSource::ImageSource> createImageSource( const std::string& filename, const std::string& directory )
{
    if( filename == "mandelbrot" )
    {
        std::vector<float4> colors = { { 1.0f, 1.0f, 1.0f, 0.0f },
                                       { 0.0f, 0.0f, 1.0f, 0.0f },
                                       { 0.0f, 0.5f, 0.0f, 0.0f },
                                       { 1.0f, 0.0f, 0.0f, 0.0f },
                                       { 1.0f, 1.0f, 0.0f, 0.0f } };
        return std::make_shared<DeviceMandelbrotImage>( 2048, 2048, /*xmin=*/-2.0, /*ymin=*/-2.0,
                                                        /*xmax=*/2.0, /*ymax=*/2.0,
                                                        /*iterations=*/512, colors );
    }
    if( filename == "constant" )
    {
        // 2K, 1K, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 = 13 colors
        std::vector<float4> mipColors = {
            { 1.0f, 1.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f, 0.0f }, { 0.0f, 0.5f, 0.0f, 0.0f },
            { 1.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 0.0f },
            { 0.0f, 0.0f, 1.0f, 0.0f }, { 0.0f, 0.5f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 0.0f },
            { 1.0f, 1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f, 0.0f },
            { 0.0f, 0.5f, 0.0f, 0.0f },
        };
        return std::make_shared<DeviceConstantImage>( 2048, 2048, mipColors );
    }
    if( filename == "multichecker" )
    {
        return std::make_shared<MultiCheckerImage<float4>>( 2048, 2048, 16, true );
    }

    return imageSource::createImageSource( filename, directory );
}

}  // namespace imageSources
