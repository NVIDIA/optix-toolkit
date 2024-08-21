// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
class GeometryResolver;
class ImageSourceFactory;
class MaterialBatch;
class MaterialResolver;
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
using GeometryResolverPtr   = std::shared_ptr<GeometryResolver>;
using ImageSourceFactoryPtr = std::shared_ptr<ImageSourceFactory>;
using ImageSourcePtr        = std::shared_ptr<imageSource::ImageSource>;
using LoggerPtr             = std::shared_ptr<otk::pbrt::Logger>;
using MaterialBatchPtr      = std::shared_ptr<MaterialBatch>;
using MaterialLoaderPtr     = std::shared_ptr<demandMaterial::MaterialLoader>;
using MaterialResolverPtr   = std::shared_ptr<MaterialResolver>;
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
