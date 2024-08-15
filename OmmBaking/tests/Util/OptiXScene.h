// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#include <OptiXToolkit/CuOmmBaking/CuBuffer.h>
#include "OptiXOmmArray.h"
#include "OptiXKernels.h"

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<void*>      RayGenSbtRecord;
typedef SbtRecord<void*>      MissSbtRecord;
typedef SbtRecord<void*>      DCSbtRecord;
typedef SbtRecord<HitSbtData> HitSbtRecord;

class OptixOmmScene
{
  public:
    OptixOmmScene(){};
    ~OptixOmmScene() { destroy(); };

    OptixOmmScene( const OptixOmmScene& source ) = delete;
    OptixOmmScene( OptixOmmScene&& source ) noexcept { swap( source ); }

    OptixOmmScene& operator=( const OptixOmmScene& source ) = delete;

    OptixOmmScene& operator=( OptixOmmScene&& source ) noexcept
    {
        swap( source );
        return *this;
    }

    OptixResult build( OptixDeviceContext context, const char* optixirInput, const size_t optixirInputSize, const OptixOmmArray& optixOmm, const cuOmmBaking::BakeInputDesc* ommBuildInput, unsigned int numBuildInputs );

    OptixResult render( uint32_t width, uint32_t height, RenderOptions options );

    const std::vector<uchar3>& getImage() const { return m_image; };

    uint32_t getErrorCount() const { return m_errorCount; }

    OptixResult destroy();

  private:
    OptixResult buildGAS( const OptixOmmArray& optixOmm, const cuOmmBaking::BakeInputDesc* ommBuildInput, unsigned int numBuildInputs );

    OptixResult buildSBT( const cuOmmBaking::BakeInputDesc* ommBuildInput, unsigned int numBuildInputs );

    OptixResult buildPipeline( const char* optixirInput, const size_t optixirInputSize );

    void swap( OptixOmmScene& source ) noexcept
    {
        std::swap( m_context, source.m_context );
        std::swap( m_pipeline, source.m_pipeline );
        std::swap( m_module, source.m_module );
        std::swap( m_sbt, source.m_sbt );
        std::swap( m_image, source.m_image );
        std::swap( m_paramsBuf, source.m_paramsBuf );
        std::swap( m_imageBuf, source.m_imageBuf );
        std::swap( m_errorCountBuf, source.m_errorCountBuf );
        std::swap( m_errorCount, source.m_errorCount );
        std::swap( m_gasHandle, source.m_gasHandle );
        std::swap( m_gas, source.m_gas );
        std::swap( m_hitSbtRecords, source.m_hitSbtRecords );
        std::swap( m_raygenSbtRecords, source.m_raygenSbtRecords );
        std::swap( m_missSbtRecords, source.m_missSbtRecords );
        std::swap( m_rgProgramGroup, source.m_rgProgramGroup );
        std::swap( m_msProgramGroup[0], source.m_msProgramGroup[0] );
        std::swap( m_msProgramGroup[1], source.m_msProgramGroup[1] );
        std::swap( m_hitgroupProgramGroup[0], source.m_hitgroupProgramGroup[0] );
        std::swap( m_hitgroupProgramGroup[1], source.m_hitgroupProgramGroup[1] );
    }

    OptixDeviceContext m_context  = {};
    OptixPipeline      m_pipeline = {};
    OptixModule        m_module   = {};
    OptixModule        m_moduleDC = {};

    OptixShaderBindingTable m_sbt = {};

    std::vector<uchar3> m_image;
    uint32_t            m_errorCount = {};

    CuBuffer<Params>   m_paramsBuf;
    CuBuffer<uchar3>   m_imageBuf;
    CuBuffer<uint32_t> m_errorCountBuf;

    OptixTraversableHandle m_gasHandle = {};
    OptixTraversableHandle m_iasHandle = {};
    CuBuffer<char>         m_gas;
    CuBuffer<char>         m_ias;

    CuBuffer<HitSbtRecord>    m_hitSbtRecords;
    CuBuffer<RayGenSbtRecord> m_raygenSbtRecords;
    CuBuffer<MissSbtRecord>   m_missSbtRecords;
    CuBuffer<DCSbtRecord>     m_dcSbtRecords;

    OptixProgramGroup m_rgProgramGroup          = {};
    OptixProgramGroup m_msProgramGroup[2]       = {};
    OptixProgramGroup m_hitgroupProgramGroup[2] = {};

    std::unordered_map<std::string, OptixProgramGroup> m_dcProgramGroupMap;
};
