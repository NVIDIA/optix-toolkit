//
// Copyright (c) 2023 NVIDIA Corporation.  All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software, related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is strictly
// prohibited.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
// AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
// INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
// SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
// LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
// BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES
//

#pragma once

#include <core/geometry.h>
#include <core/paramset.h>
#include <core/transform.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace otk {
namespace pbrt {

class MeshLoader;
using MeshLoaderPtr = std::shared_ptr<MeshLoader>;

struct PerspectiveCameraDefinition
{
    float             fov;            // field of view in Y axis
    float             focalDistance;  //
    float             lensRadius;     //
    ::pbrt::Transform cameraToWorld;  // camera to world space transform
    ::pbrt::Transform cameraToScreen; // camera to screen space transform
};

struct DistantLightDefinition
{
    ::pbrt::Point3f   scale;         // Modulates the radiance emitted from the light source.
    ::pbrt::Point3f   color;         // Simplified radiance emitted from the light source.
    ::pbrt::Vector3f  direction;     // Light direction defined by two points.
    ::pbrt::Transform lightToWorld;  // Current transform at the time the light was defined.
};

using DistantLightList = std::vector<DistantLightDefinition>;

struct InfiniteLightDefinition
{
    ::pbrt::Point3f   scale;               // Modulates the radiance emitted from the light source.
    ::pbrt::Point3f   color;               // Simplified radiance emitted from the light source.
    int               shadowSamples;       // Number of suggested shadow samples to take.
    std::string       environmentMapName;  // Empty string means constant color.
    ::pbrt::Transform lightToWorld;        // Current transform at the time the light was defined.
};

using InfiniteLightList = std::vector<InfiniteLightDefinition>;

struct PlasticMaterial
{
    ::pbrt::Point3f Ka;
    ::pbrt::Point3f Kd;
    ::pbrt::Point3f Ks;
    std::string     alphaMapFileName;
    std::string     diffuseMapFileName;
    std::string     specularMapFileName;
};

struct PlyMeshData
{
    std::string   fileName;
    MeshLoaderPtr loader;
};

struct TriangleMeshData
{
    std::vector<int>             indices;
    std::vector<::pbrt::Point3f> points;
    std::vector<::pbrt::Point3f> normals;
    std::vector<::pbrt::Point2f> uvs;
};

struct SphereData
{
    float radius;
    float zMin;
    float zMax;
    float phiMax;
};

struct ShapeDefinition
{
    std::string       type;          // "plymesh", "trianglemesh" or "sphere"
    ::pbrt::Transform transform;     // object to world space transformation
    PlasticMaterial   material;      //
    ::pbrt::Bounds3f  bounds;        // object space bounds
    PlyMeshData       plyMesh;       //
    TriangleMeshData  triangleMesh;  //
    SphereData        sphere;        //
};

using ShapeList = std::vector<ShapeDefinition>;

struct ObjectDefinition
{
    ::pbrt::Transform transform;  // object to world space transform
    ::pbrt::Bounds3f  bounds;     // object space bounds of transformed shapes in this object
};

// ObjectInstance results in the name of an object and its instance transform
struct ObjectInstanceDefinition
{
    std::string       name;       // name of the instantiated object
    ::pbrt::Transform transform;  // object instance to world space instance transform
    ::pbrt::Bounds3f  bounds;     // object space bounds of object
};

using ObjectInstanceList     = std::vector<ObjectInstanceDefinition>;
using ObjectInstanceCountMap = std::map<std::string, unsigned int>;
using ObjectMap              = std::map<std::string, ObjectDefinition>;
using ObjectShapeMap         = std::map<std::string, ShapeList>;

struct LookAtDefinition
{
    ::pbrt::Point3f  eye;
    ::pbrt::Point3f  lookAt;
    ::pbrt::Vector3f up;
};

struct SceneDescription
{
    int                         warnings;         // number of warnings found during parse
    int                         errors;           // number of errors found during parse
    LookAtDefinition            lookAt;           // last parameters to LookAt
    PerspectiveCameraDefinition camera;           // camera parameters
    DistantLightList            distantLights;    // distant light definitions
    InfiniteLightList           infiniteLights;   // infinite light definitions
    ::pbrt::Bounds3f            bounds;           // world space bounds of all object instances and free shapes
    ObjectMap                   objects;          // map of object names to object definitions
    ShapeList                   freeShapes;       // vector of shapes not in any object
    ObjectInstanceCountMap      instanceCounts;   // map of object names to object instance counts
    ObjectInstanceList          objectInstances;  // vector of object instance definitions
    ObjectShapeMap              objectShapes;     // map of object names to vector of shapes
};

using SceneDescriptionPtr = std::shared_ptr<SceneDescription>;

}  // namespace pbrt
}  // namespace otk
