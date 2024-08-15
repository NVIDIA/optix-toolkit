// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ImageSource/ImageSourceCache.h>

namespace imageSource {

std::shared_ptr<ImageSource> ImageSourceCache::get( const std::string& path )
{
    // Use a cached ImageSource if possible.
    std::shared_ptr<ImageSource> imageSource{ find( path ) };
    if( imageSource )
        return imageSource;

    // Create a new ImageSource and cache it.
    imageSource = createImageSource( path );
    m_cache[path] = imageSource;
    return imageSource;
}

CacheStatistics ImageSourceCache::getStatistics() const
{
    CacheStatistics result{};
    for (const auto &keyValue :  m_cache)
    {
        ++result.numImageSources;
        result.totalBytesRead += keyValue.second->getNumBytesRead();
        result.totalTilesRead += keyValue.second->getNumTilesRead();
        result.totalReadTime += keyValue.second->getTotalReadTime();
    }
    return result;
}

}  // namespace imageSource
