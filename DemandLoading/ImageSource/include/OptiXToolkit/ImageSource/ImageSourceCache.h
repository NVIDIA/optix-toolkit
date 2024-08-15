// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file ImageSourceCache.h
/// Cache for ImageSource instances.

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/ImageSourceCacheStatistics.h>

#include <map>
#include <memory>
#include <string>

namespace imageSource {

/// Cache for ImageSource instances.
class ImageSourceCache
{
  public:
    /// Returns the image source from the cache associated with the given path.
    /// If no such image source exists, returns an empty shared_ptr.
    std::shared_ptr<ImageSource> find( const std::string& path ) const
    {
        auto it = m_cache.find( path );
        return it == m_cache.end() ? std::shared_ptr<ImageSource>() : it->second;
    }

    /// Get the specified ImageSource.  Returns a cached ImageSource if possible; otherwise a new
    /// instance is created.  The type of the ImageSource is determined by the filename extension.
    /// Returns the TextureInfo via result parameter.  Throws an exception on error.
    std::shared_ptr<ImageSource> get( const std::string& path );

    /// Set the specified ImageSource to be associated with the given path.  This allows the application
    /// to insert their own ImageSources into the cache without relying on createImageSource to create
    /// the image from the associated filename.  For instance, this allows tiled or mipmap adapted
    /// images to be inserted into the cache.
    void set( const std::string& path, const std::shared_ptr<ImageSource>& image ) { m_cache[path] = image; }

    /// Return aggregate statistics for all ImageSources in the cache
    CacheStatistics getStatistics() const;

private:
    std::map<std::string, std::shared_ptr<ImageSource>> m_cache;
};

}  // namespace imageSource
