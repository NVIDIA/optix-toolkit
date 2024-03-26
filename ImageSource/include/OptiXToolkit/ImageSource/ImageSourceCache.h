//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

/// \file ImageSourceCache.h
/// Cache for ImageSource instances.

#include <OptiXToolkit/ImageSource/ImageSource.h>

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
        auto it = m_map.find( path );
        return it == m_map.end() ? std::shared_ptr<ImageSource>() : it->second;
    }

    /// Get the specified ImageSource.  Returns a cached ImageSource if possible; otherwise a new
    /// instance is created.  The type of the ImageSource is determined by the filename extension.
    /// Returns the TextureInfo via result parameter.  Throws an exception on error.
    std::shared_ptr<ImageSource> get( const std::string& path );

    /// Set the specified ImageSource to be associated with the given path.  This allows the application
    /// to insert their own ImageSources into the cache without relying on createImageSource to create
    /// the image from the associated filename.  For instance, this allows tiled or mipmap adapted
    /// images to be inserted into the cache.
    void set( const std::string& path, const std::shared_ptr<ImageSource>& image ) { m_map[path] = image; }

private:
    std::map<std::string, std::shared_ptr<ImageSource>> m_map;
};

}  // namespace imageSource
