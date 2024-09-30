// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Dependencies.h"
#include "DemandPbrtScene/GeometryCacheStatistics.h"
#include "DemandPbrtScene/Params.h"
#include "DemandPbrtScene/Primitive.h"

#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <cuda.h>

#include <optix.h>

#include <memory>
#include <vector>

namespace demandPbrtScene {

struct GeometryCacheEntry
{
    CUdeviceptr            accelBuffer;  // device pointer to geometry acceleration structure
    OptixTraversableHandle traversable;  // traversable handle for this GAS
    GeometryPrimitive      primitive;    // primitive type stored in this GAS
    TriangleNormals*       devNormals;   // device pointer to per-primitive normals, nullptr if none
    TriangleUVs*           devUVs;       // device pointer to per-primitive texture coordinates, nullptr if none
    std::vector<uint_t>    primitiveGroupEndIndices;  // primitive indices that end each group
};

class GeometryCache
{
  public:
    virtual ~GeometryCache() = default;

    virtual GeometryCacheEntry getShape( OptixDeviceContext context, CUstream stream, const otk::pbrt::ShapeDefinition& shape ) = 0;

    virtual GeometryCacheEntry getObject( OptixDeviceContext                 context,
                                          CUstream                           stream,
                                          const otk::pbrt::ObjectDefinition& object,
                                          const otk::pbrt::ShapeList&        shapes,
                                          GeometryPrimitive                  primitive,
                                          MaterialFlags                      flags ) = 0;

    virtual GeometryCacheStatistics getStatistics() const = 0;
};

class FileSystemInfo
{
  public:
    virtual ~FileSystemInfo() = default;

    virtual unsigned long long getSize( const std::string& path ) const = 0;
};

using FileSystemInfoPtr = std::shared_ptr<FileSystemInfo>;

FileSystemInfoPtr createFileSystemInfo();

GeometryCachePtr createGeometryCache( FileSystemInfoPtr fileSystemInfo );

}  // namespace demandPbrtScene
