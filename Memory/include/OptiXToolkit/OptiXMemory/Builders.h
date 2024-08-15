// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <optix.h>

#include <cuda.h>

#include <stdexcept>

namespace otk {

/// Build an OptixProgramGroupDesc array in a declarative fashion.
///
/// Instantiate this class and chain methods to build up successive entries
/// in the array of program group descriptions.  Each chained method fills in
/// one of the entries in the fixed sized array.  If too many methods are
/// called for the size of the supplied array, a std::runtime_error exception
/// is thrown.
///
/// This class does not own the array of program group description structures.
///
class ProgramGroupDescBuilder
{
  public:
    /// Constructor
    ///
    /// @param descs Fixed-length array of OptixProgramGroupDesc structures.
    /// @param module OptixModule to use for the source of entry points.
    ///
    template <int N>
    ProgramGroupDescBuilder( OptixProgramGroupDesc ( &descs )[N], OptixModule module )
        : m_descs( descs )
        , m_size( N )
        , m_module( module )
    {
    }

    /// Build a ray generating program group description.
    ///
    /// @param entryPoint The entry function name for the raygen program.
    ///
    ProgramGroupDescBuilder& raygen( const char* entryPoint )
    {
        checkOverflow();
        current().kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        current().raygen.module            = m_module;
        current().raygen.entryFunctionName = entryPoint;
        return next();
    }
    ProgramGroupDescBuilder& raygen( OptixProgramGroupSingleModule entryPoint )
    {
        checkOverflow();
        current().kind   = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        current().raygen = entryPoint;
        return next();
    }

    /// Build a miss program group description.
    ///
    /// @param entryPoint The entry function name for the miss program.
    ///
    ProgramGroupDescBuilder& miss( const char* entryPoint )
    {
        checkOverflow();
        current().kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        current().miss.module            = m_module;
        current().miss.entryFunctionName = entryPoint;
        return next();
    }
    ProgramGroupDescBuilder& miss( OptixProgramGroupSingleModule entryPoint )
    {
        checkOverflow();
        current().kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        current().miss = entryPoint;
        return next();
    }

    /// Build a hit group program group description for a closest hit program from another module.
    ///
    /// @param module The OptiX module for the closest hit program.
    /// @param closestHit The entry function name for the closest hit program.
    ///
    ProgramGroupDescBuilder& hitGroupCH( OptixModule module, const char* closestHit )
    {
        checkOverflow();
        current().kind                      = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        OptixProgramGroupHitgroup& hitGroup = current().hitgroup;

        hitGroup.moduleIS            = nullptr;
        hitGroup.entryFunctionNameIS = nullptr;
        hitGroup.moduleAH            = nullptr;
        hitGroup.entryFunctionNameAH = nullptr;
        hitGroup.moduleCH            = module;
        hitGroup.entryFunctionNameCH = closestHit;
        return next();
    }
    ProgramGroupDescBuilder& hitGroupCH( OptixProgramGroupSingleModule entryPoint )
    {
        return hitGroupCH( entryPoint.module, entryPoint.entryFunctionName );
    }

    /// Build a hit group program group description for intersection and closest hit programs in the builder's module.
    ///
    /// @param closestHit The entry function name for the closest hit program.
    /// @param intersection The entry function name for the intersection program.
    ///
    ProgramGroupDescBuilder& hitGroupISCH( const char* intersection, const char* closestHit )
    {
        return hitGroupISCH( m_module, intersection, m_module, closestHit );
    }

    /// Build a hit group program group description for intersection and closest hit programs in separate modules.
    ///
    /// @param chModule The module containing the closest hit program.
    /// @param closestHit The entry function name for the closest hit program.
    /// @param isModule The module containing the intersection program.
    /// @param intersection The entry function name for the intersection program.
    ///
    ProgramGroupDescBuilder& hitGroupISCH( OptixModule isModule, const char* intersection, OptixModule chModule, const char* closestHit )
    {
        checkOverflow();
        current().kind                      = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        OptixProgramGroupHitgroup& hitGroup = current().hitgroup;

        hitGroup.moduleIS            = isModule;
        hitGroup.entryFunctionNameIS = intersection;
        hitGroup.moduleAH            = nullptr;
        hitGroup.entryFunctionNameAH = nullptr;
        hitGroup.moduleCH            = chModule;
        hitGroup.entryFunctionNameCH = closestHit;
        return next();
    }
    ProgramGroupDescBuilder& hitGroupISCH( OptixProgramGroupSingleModule intersect, OptixProgramGroupSingleModule closestHit )
    {
        checkOverflow();
        current().kind                      = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        OptixProgramGroupHitgroup& hitGroup = current().hitgroup;
        hitGroup.moduleIS                   = intersect.module;
        hitGroup.entryFunctionNameIS        = intersect.entryFunctionName;
        hitGroup.moduleAH                   = nullptr;
        hitGroup.entryFunctionNameAH        = nullptr;
        hitGroup.moduleCH                   = closestHit.module;
        hitGroup.entryFunctionNameCH        = closestHit.entryFunctionName;
        return next();
    }

    /// Build a hit group program group description for intersection, any hit and closest hit programs in separate modules.
    ///
    /// @param isModule     The module containing the intersection program.
    /// @param intersection The entry function name for the intersection program.
    /// @param ahModule     The module containing the closest hit program.
    /// @param anyHit       The entry function name for the closest hit program.
    /// @param chModule     The module containing the closest hit program.
    /// @param closestHit   The entry function name for the closest hit program.
    ///
    ProgramGroupDescBuilder& hitGroupISAHCH( OptixModule isModule,
                                             const char* intersection,
                                             OptixModule ahModule,
                                             const char* anyHit,
                                             OptixModule chModule,
                                             const char* closestHit )
    {
        checkOverflow();
        current().kind                      = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        OptixProgramGroupHitgroup& hitGroup = current().hitgroup;

        hitGroup.moduleIS            = isModule;
        hitGroup.entryFunctionNameIS = intersection;
        hitGroup.moduleAH            = ahModule;
        hitGroup.entryFunctionNameAH = anyHit;
        hitGroup.moduleCH            = chModule;
        hitGroup.entryFunctionNameCH = closestHit;
        return next();
    }
    ProgramGroupDescBuilder& hitGroupISAHCH( OptixProgramGroupSingleModule intersect,
                                             OptixProgramGroupSingleModule anyHit,
                                             OptixProgramGroupSingleModule closestHit )
    {
        return hitGroupISAHCH( intersect.module, intersect.entryFunctionName, anyHit.module, anyHit.entryFunctionName,
                               closestHit.module, closestHit.entryFunctionName );
    }

  private:
    OptixProgramGroupDesc& current() { return m_descs[m_current]; }

    void checkOverflow()
    {
        if( m_current == m_size )
            throw std::runtime_error( "OptixProgramGroupDesc array is full" );
    }

    ProgramGroupDescBuilder& next()
    {
        ++m_current;
        return *this;
    }

    OptixProgramGroupDesc* m_descs;
    int                    m_size;
    OptixModule            m_module;
    int                    m_current{};
};

/// Build an OptixBuildInput array in a declarative fashion.
///
/// Instantiate this class and chain methods to build up successive entries
/// in the array of build input descriptions.  Each chained method fills in
/// one of the entries in the fixed sized array.  If too many methods are
/// called for the size of the supplied array, a std::runtime_error exception
/// is thrown.
///
/// This class does not own the array of build input description structures.
///
class BuildInputBuilder
{
  public:
    using uint_t = unsigned int;

    /// Constructor
    ///
    /// @param buildInputs Fixed-length array of OptixBuildInput structures.
    ///
    template <size_t N>
    BuildInputBuilder( OptixBuildInput ( &buildInputs )[N] )
        : m_buildInputs( buildInputs )
        , m_size( N )
    {
    }

    BuildInputBuilder( OptixBuildInput* buildInputs, size_t numInputs )
        : m_buildInputs( buildInputs )
        , m_size( numInputs )
    {
    }

    /// Describe a triangle array build input.
    ///
    BuildInputBuilder& triangles( unsigned int       numVertices,
                                  const CUdeviceptr* vertices,
                                  OptixVertexFormat  vertexFormat,
                                  unsigned int       numTriangles,
                                  const CUdeviceptr  indices,
                                  OptixIndicesFormat indexFormat,
                                  const uint32_t*    flags,
                                  unsigned int       numSbtRecords )
    {
        checkOverflow();
        current().type                          = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        OptixBuildInputTriangleArray& triangles = current().triangleArray;
        triangles.vertexBuffers                 = vertices;
        triangles.numVertices                   = numVertices;
        triangles.vertexFormat                  = vertexFormat;
        triangles.vertexStrideInBytes           = 0;
        triangles.indexBuffer                   = indices;
        triangles.numIndexTriplets              = numTriangles;
        triangles.indexFormat                   = indexFormat;
        triangles.indexStrideInBytes            = 0;
        triangles.preTransform                  = CUdeviceptr{};
        triangles.flags                         = flags;
        triangles.numSbtRecords                 = numSbtRecords;
        triangles.sbtIndexOffsetBuffer          = 0;
        triangles.sbtIndexOffsetSizeInBytes     = 0;
        triangles.sbtIndexOffsetStrideInBytes   = 0;
        triangles.primitiveIndexOffset          = 0;
        triangles.transformFormat               = OPTIX_TRANSFORM_FORMAT_NONE;
#if OPTIX_VERSION >= 70600
        triangles.opacityMicromap = OptixBuildInputOpacityMicromap{};
#endif
        return next();
    }

    /// Describe a custom primitive build input.
    ///
    /// The arguments match the members of the customPrimitiveArray member
    /// of the discriminated union that is OptixBuildInput.
    ///
    /// @param aabbBuffers Points to host array of device pointers to AABBs (type OptixAabb), one per motion step.
    /// @param aabbInputFlags Array of flags to specify flags per sbt record.
    /// @param numSbtRecords Number of sbt records available to the sbt index offset override.
    /// @param numPrimitives Number of primitives in each buffer.
    /// @param devSbtIndexOffsetBuffer Device pointer to per-primitive local sbt index offset buffer. May be nullptr.
    /// @param sbtIndexOffsetSizeInBytes Size of type of the sbt index offset. Needs to be 0, 1, 2 or 4 (8, 16 or 32 bit).
    ///
    BuildInputBuilder& customPrimitives( void**        aabbBuffers,
                                         const uint_t* aabbInputFlags,
                                         uint_t        numSbtRecords,
                                         uint_t        numPrimitives,
                                         void*         devSbtIndexOffsetBuffer,
                                         uint_t        sbtIndexOffsetSizeInBytes )
    {
        checkOverflow();
        current().type                                      = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        current().customPrimitiveArray.aabbBuffers          = reinterpret_cast<CUdeviceptr*>( aabbBuffers );
        current().customPrimitiveArray.flags                = aabbInputFlags;
        current().customPrimitiveArray.numSbtRecords        = numSbtRecords;
        current().customPrimitiveArray.numPrimitives        = numPrimitives;
        current().customPrimitiveArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>( devSbtIndexOffsetBuffer );
        current().customPrimitiveArray.sbtIndexOffsetSizeInBytes = sbtIndexOffsetSizeInBytes;
        return next();
    }

    /// Describe an array of OptixInstance structures build input.
    ///
    /// The arguments match the members of the instanceArray member
    /// of the discriminated union that is OptixBuildInput.
    ///
    /// @param instances Points to device array of OptixInstance structures.
    /// @param numInstances Number of OptixInstance structures in the device array.
    ///
    BuildInputBuilder& instanceArray( CUdeviceptr instances, unsigned int numInstances )
    {
        checkOverflow();
        current().type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        current().instanceArray.instances    = numInstances ? instances : 0;
        current().instanceArray.numInstances = numInstances;
        return next();
    }

    /// Describe an array of pointers to OptixInstance structures build input.
    ///
    /// The arguments match the members of the instanceArray member
    /// of the discriminated union that is OptixBuildInput.
    ///
    /// @param instancePtrs Points to device array of OptixInstance structure pointers.
    /// @param numInstances Number of OptixInstance structure pointers in the device array.
    ///
    BuildInputBuilder& instancePtrArray( CUdeviceptr instancePtrs, unsigned int numInstances )
    {
        checkOverflow();
        current().type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS;
        current().instanceArray.instances    = instancePtrs;
        current().instanceArray.numInstances = numInstances;
        return next();
    }

#if OPTIX_VERSION >= 70500
    BuildInputBuilder& spheres( const CUdeviceptr* vertices, unsigned int numSpheres, const CUdeviceptr* radii, const uint32_t* flags, unsigned int numSbtRecords )
    {
        checkOverflow();
        current().type                      = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        OptixBuildInputSphereArray& spheres = current().sphereArray;

        spheres.vertexBuffers               = numSpheres ? vertices : nullptr;
        spheres.vertexStrideInBytes         = 0;
        spheres.numVertices                 = numSpheres;
        spheres.radiusBuffers               = numSpheres ? radii : nullptr;
        spheres.radiusStrideInBytes         = 0;
        spheres.singleRadius                = 0;
        spheres.flags                       = flags;
        spheres.numSbtRecords               = numSbtRecords;
        spheres.sbtIndexOffsetBuffer        = 0;
        spheres.sbtIndexOffsetSizeInBytes   = 0;
        spheres.sbtIndexOffsetStrideInBytes = 0;
        spheres.primitiveIndexOffset        = 0;
        return next();
    }

    BuildInputBuilder& spheres( const CUdeviceptr* vertices,
                                unsigned int       numSpheres,
                                const CUdeviceptr* radii,
                                const uint32_t*    flags,
                                unsigned int       numSbtRecords,
                                void*              devSbtIndexOffsetBuffer,
                                uint_t             sbtIndexOffsetSizeInBytes )
    {
        checkOverflow();
        current().type                      = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        OptixBuildInputSphereArray& spheres = current().sphereArray;

        spheres.vertexBuffers        = numSpheres ? vertices : nullptr;
        spheres.vertexStrideInBytes  = 0;
        spheres.numVertices          = numSpheres;
        spheres.radiusBuffers        = numSpheres ? radii : nullptr;
        spheres.radiusStrideInBytes  = 0;
        spheres.singleRadius         = 0;
        spheres.flags                = flags;
        spheres.numSbtRecords        = numSbtRecords;
        spheres.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>( devSbtIndexOffsetBuffer );

        spheres.sbtIndexOffsetSizeInBytes   = sbtIndexOffsetSizeInBytes;
        spheres.sbtIndexOffsetStrideInBytes = 0;
        spheres.primitiveIndexOffset        = 0;
        return next();
    }
#endif  // OPTIX_VERSION >= 70500

  private:
    OptixBuildInput& current() { return m_buildInputs[m_current]; }

    void checkOverflow()
    {
        if( m_current == m_size )
            throw std::runtime_error( "OptixBuildInput array is full" );
    }

    BuildInputBuilder& next()
    {
        ++m_current;
        return *this;
    }

    OptixBuildInput* m_buildInputs;
    size_t           m_size;
    size_t           m_current{};
};

}  // namespace otk
