//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>

namespace otk {
namespace pbrt {
struct SceneDescription;
class Logger;
class MeshInfoReader;
class SceneLoader;
}  // namespace pbrt
}  // namespace otk

namespace demandGeometry {
class GeometryLoader;
}  // namespace demandGeometry

namespace demandLoading {
class DemandLoader;
}  // namespace demandLoading

namespace demandMaterial {
class MaterialLoader;
}  // namespace demandMaterial

namespace imageSource {
class ImageSource;
}  // namespace imageSource

namespace demandPbrtScene {

struct Options;
struct Params;
class DemandTextureCache;
class GeometryCache;
class ImageSourceFactory;
class ProgramGroups;
class ProxyFactory;
class Renderer;
class Scene;
class SceneProxy;
class UserInterface;

using uint_t = unsigned int;

using DemandTextureCachePtr = std::shared_ptr<DemandTextureCache>;
using DemandLoaderPtr       = std::shared_ptr<demandLoading::DemandLoader>;
using GeometryCachePtr      = std::shared_ptr<GeometryCache>;
using GeometryLoaderPtr     = std::shared_ptr<demandGeometry::GeometryLoader>;
using ImageSourceFactoryPtr = std::shared_ptr<ImageSourceFactory>;
using ImageSourcePtr        = std::shared_ptr<imageSource::ImageSource>;
using LoggerPtr             = std::shared_ptr<otk::pbrt::Logger>;
using MaterialLoaderPtr     = std::shared_ptr<demandMaterial::MaterialLoader>;
using MeshInfoReaderPtr     = std::shared_ptr<otk::pbrt::MeshInfoReader>;
using PbrtSceneLoaderPtr    = std::shared_ptr<otk::pbrt::SceneLoader>;
using ProgramGroupsPtr      = std::shared_ptr<ProgramGroups>;
using ProxyFactoryPtr       = std::shared_ptr<ProxyFactory>;
using RendererPtr           = std::shared_ptr<Renderer>;
using SceneDescriptionPtr   = std::shared_ptr<::otk::pbrt::SceneDescription>;
using SceneProxyPtr         = std::shared_ptr<SceneProxy>;
using ScenePtr              = std::shared_ptr<Scene>;
using UserInterfacePtr      = std::shared_ptr<UserInterface>;

}  // namespace demandPbrtScene
