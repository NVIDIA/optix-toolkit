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

#include "Dependencies.h"
#include "GeometryCacheStatistics.h"
#include "Params.h"
#include "Primitive.h"

#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <cuda.h>

#include <optix.h>

#include <memory>

namespace demandPbrtScene {

struct GeometryCacheEntry
{
    CUdeviceptr            accelBuffer;  // device pointer to geometry acceleration structure
    OptixTraversableHandle traversable;  // traversable handle for this GAS
    GeometryPrimitive      primitive;    // primitive type stored in this GAS
    TriangleNormals*       devNormals;   // device pointer to per-primitive normals, nullptr if none
    TriangleUVs*           devUVs;       // device pointer to per-primitive texture coordinates, nullptr if none
};

class GeometryCache
{
  public:
    virtual ~GeometryCache() = default;

    virtual GeometryCacheEntry getShape( OptixDeviceContext context, CUstream stream, const otk::pbrt::ShapeDefinition& shape ) = 0;

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
