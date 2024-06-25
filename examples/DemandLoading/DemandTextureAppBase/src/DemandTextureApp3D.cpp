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

// This include is needed to avoid a link error
#include <optix_stubs.h>
#include <OptiXToolkit/DemandTextureAppBase/DemandTextureApp3D.h>

using namespace demandTextureApp;


DemandTextureApp3D::DemandTextureApp3D( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
    : DemandTextureApp( appTitle, width, height, outFileName, glInterop )
{
}


void DemandTextureApp3D::buildAccel( PerDeviceOptixState& state )
{
    // Copy vertex data to device
    void* d_vertices = nullptr;
    const size_t vertices_size_bytes = m_vertices.size() * sizeof( float4 );
    OTK_ERROR_CHECK( cudaMalloc( &d_vertices, vertices_size_bytes ) );
    OTK_ERROR_CHECK( cudaMemcpy( d_vertices, m_vertices.data(), vertices_size_bytes, cudaMemcpyHostToDevice ) );
    state.d_vertices = reinterpret_cast<CUdeviceptr>( d_vertices );

    // Copy material indices to device
    void* d_material_indices = nullptr;
    const size_t material_indices_size_bytes = m_material_indices.size() * sizeof( uint32_t );
    OTK_ERROR_CHECK( cudaMalloc( &d_material_indices, material_indices_size_bytes ) );
    OTK_ERROR_CHECK( cudaMemcpy( d_material_indices, m_material_indices.data(), material_indices_size_bytes, cudaMemcpyHostToDevice ) );

    // Make triangle input flags (one per sbt record).  Here, we are just disabling the anyHit programs
    std::vector<uint32_t> triangle_input_flags( m_materials.size(), OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT );

    // Make GAS accel build inputs
    OptixBuildInput triangle_input                           = {};
    triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3; 
    triangle_input.triangleArray.vertexStrideInBytes         = static_cast<uint32_t>( sizeof( float4 ) );
    triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>( m_vertices.size() );
    triangle_input.triangleArray.vertexBuffers               = &state.d_vertices;
    triangle_input.triangleArray.flags                       = triangle_input_flags.data();
    triangle_input.triangleArray.numSbtRecords               = static_cast<uint32_t>( triangle_input_flags.size() );
    triangle_input.triangleArray.sbtIndexOffsetBuffer        = reinterpret_cast<CUdeviceptr>( d_material_indices );
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );

    // Make accel options
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    // Compute memory usage for accel build
    OptixAccelBufferSizes gas_buffer_sizes;
    const unsigned int num_build_inputs = 1;
    OTK_ERROR_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &triangle_input, num_build_inputs, &gas_buffer_sizes ) );

    // Allocate temporary buffer needed for accel build
    void* d_temp_buffer = nullptr;
    OTK_ERROR_CHECK( cudaMalloc( &d_temp_buffer, gas_buffer_sizes.tempSizeInBytes ) );

    // Allocate output buffer for (non-compacted) accel build result, and also compactedSize property.
    void* d_buffer_temp_output_gas_and_compacted_size = nullptr;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    OTK_ERROR_CHECK( cudaMalloc( &d_buffer_temp_output_gas_and_compacted_size, compactedSizeOffset + 8 ) );

    // Set up the accel build to return the compacted size, so compaction can be run after the build
    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result             = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    // Finally perform the accel build
    OTK_ERROR_CHECK( optixAccelBuild(
                state.context,
                CUstream{0},
                &accel_options,
                &triangle_input,
                num_build_inputs,                    
                reinterpret_cast<CUdeviceptr>( d_temp_buffer ),
                gas_buffer_sizes.tempSizeInBytes,
                reinterpret_cast<CUdeviceptr>( d_buffer_temp_output_gas_and_compacted_size ),
                gas_buffer_sizes.outputSizeInBytes,
                &state.gas_handle,
                &emitProperty,
                1
                ) );

    // Delete temporary buffers used for the accel build
    OTK_ERROR_CHECK( cudaFree( d_temp_buffer ) );
    OTK_ERROR_CHECK( cudaFree( d_material_indices ) );

    // Copy the size of the compacted GAS accel back from the device
    size_t compacted_gas_size;
    OTK_ERROR_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    // If compaction reduces the size of the accel, copy to a new buffer and delete the old one
    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        OTK_ERROR_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_gas_output_buffer ), compacted_gas_size ) );
        // use handle as input and output
        OTK_ERROR_CHECK( optixAccelCompact( state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle ) );
        OTK_ERROR_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        state.d_gas_output_buffer = reinterpret_cast<CUdeviceptr>( d_buffer_temp_output_gas_and_compacted_size );
    }
}


void  DemandTextureApp3D::createSBT( PerDeviceOptixState& state )
{
    // Raygen record 
    void*  d_raygen_record = nullptr;
    const size_t raygen_record_size = sizeof( RayGenSbtRecord );
    OTK_ERROR_CHECK( cudaMalloc( &d_raygen_record, raygen_record_size ) );
    RayGenSbtRecord raygen_record = {};
    OTK_ERROR_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &raygen_record ) );
    OTK_ERROR_CHECK( cudaMemcpy( d_raygen_record, &raygen_record, raygen_record_size, cudaMemcpyHostToDevice ) );

    // Miss record
    void* d_miss_record = nullptr;
    const size_t miss_record_size = sizeof( MissSbtRecord );
    OTK_ERROR_CHECK( cudaMalloc( &d_miss_record, miss_record_size ) );
    MissSbtRecord miss_record;
    OTK_ERROR_CHECK( optixSbtRecordPackHeader( state.miss_prog_group, &miss_record ) );
    miss_record.data.background_color = m_backgroundColor;
    OTK_ERROR_CHECK( cudaMemcpy( d_miss_record, &miss_record, miss_record_size, cudaMemcpyHostToDevice ) );

    // Hitgroup records (one for each material)
    const unsigned int MAT_COUNT = static_cast<unsigned int>( m_materials.size() );
    void* d_hitgroup_records = nullptr;
    const size_t hitgroup_record_size = sizeof( TriangleHitGroupSbtRecord );
    OTK_ERROR_CHECK( cudaMalloc( &d_hitgroup_records, hitgroup_record_size * MAT_COUNT ) );
    std::vector<TriangleHitGroupSbtRecord> hitgroup_records( MAT_COUNT );
    for( unsigned int mat_idx = 0; mat_idx < MAT_COUNT; ++mat_idx )
    {
        OTK_ERROR_CHECK( optixSbtRecordPackHeader( state.hitgroup_prog_group, &hitgroup_records[mat_idx] ) );
        TriangleHitGroupData* hg_data = &hitgroup_records[mat_idx].data;
        // Copy material definition, and then fill in device-specific values for vertices, normals, tex_coords
        *hg_data = m_materials[mat_idx];
        hg_data->vertices = reinterpret_cast<float4*>( state.d_vertices );
        hg_data->normals = state.d_normals;
        hg_data->tex_coords = state.d_tex_coords;
    }
    OTK_ERROR_CHECK( cudaMemcpy( d_hitgroup_records, &hitgroup_records[0], hitgroup_record_size * MAT_COUNT, cudaMemcpyHostToDevice ) );

    // Set up SBT
    state.sbt.raygenRecord                = reinterpret_cast<CUdeviceptr>( d_raygen_record );
    state.sbt.missRecordBase              = reinterpret_cast<CUdeviceptr>( d_miss_record );
    state.sbt.missRecordStrideInBytes     = static_cast<uint32_t>( miss_record_size );
    state.sbt.missRecordCount             = 1;
    state.sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>( d_hitgroup_records );
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    state.sbt.hitgroupRecordCount         = MAT_COUNT;
}


SurfaceTexture DemandTextureApp3D::makeSurfaceTex( int kd, int kdtex, int ks, int kstex, int kt, int kttex, float roughness, float ior )
{
    SurfaceTexture tex;
    tex.emission     = ColorTex{ float3{ 0.0f, 0.0f, 0.0f }, -1 };
    tex.diffuse      = ColorTex{ float3{ ((kd>>16)&0xff)/255.0f, ((kd>>8)&0xff)/255.0f, ((kd>>0)&0xff)/255.0f }, kdtex };
    tex.specular     = ColorTex{ float3{ ((ks>>16)&0xff)/255.0f, ((ks>>8)&0xff)/255.0f, ((ks>>0)&0xff)/255.0f }, kstex };
    tex.transmission = ColorTex{ float3{ ((kt>>16)&0xff)/255.0f, ((kt>>8)&0xff)/255.0f, ((kt>>0)&0xff)/255.0f }, kttex };
    tex.roughness    = roughness;
    tex.ior          = ior;
    return tex;
}


void DemandTextureApp3D::addShapeToScene( std::vector<Vert>& shape, unsigned int materialId )
{
    for( unsigned int i=0; i<shape.size(); ++i )
    {
        m_vertices.push_back( make_float4( shape[i].p ) );
        m_normals.push_back( shape[i].n );
        m_tex_coords.push_back( shape[i].t );
        if( i % 3 == 0 )
            m_material_indices.push_back( materialId );
    }
}


void DemandTextureApp3D::copyGeometryToDevice()
{
    for( PerDeviceOptixState& state : m_perDeviceOptixStates )
    {
        OTK_ERROR_CHECK( cudaSetDevice( state.device_idx ) );
        
        // m_vertices copied in buildAccel
        // m_material_indices copied in createSBT
        // m_materials copied in createSBT

        // m_normals
        OTK_ERROR_CHECK( cudaMalloc( &state.d_normals, m_normals.size() * sizeof(float3) ) );
        OTK_ERROR_CHECK( cudaMemcpy( state.d_normals, m_normals.data(),  m_normals.size() * sizeof(float3), cudaMemcpyHostToDevice ) );

        // m_tex_coords
        OTK_ERROR_CHECK( cudaMalloc( &state.d_tex_coords, m_tex_coords.size() * sizeof(float2) ) );
        OTK_ERROR_CHECK( cudaMemcpy( state.d_tex_coords, m_tex_coords.data(),  m_tex_coords.size() * sizeof(float2), cudaMemcpyHostToDevice ) );
    }
}


void DemandTextureApp3D::cursorPosCallback( GLFWwindow* /*window*/, double xpos, double ypos )
{
    if( m_mouseButton < 0 )
        return;

    const float pan = 0.03f;
    const float rot = 0.002f;

    float3 U, V, W;
    m_camera.UVWFrame( U, V, W );
    V.z = 0.0f;
    float dx = static_cast<float>( xpos - m_mousePrevX );
    float dy = static_cast<float>( ypos - m_mousePrevY );

    if( m_mouseButton == GLFW_MOUSE_BUTTON_LEFT )  
        panCamera( ( pan * dx * normalize(U) ) + ( -pan * dy * normalize(V) ) );
    else if( m_mouseButton == GLFW_MOUSE_BUTTON_RIGHT )  
        rotateCamera( -rot * dx );

    m_mousePrevX = xpos;
    m_mousePrevY = ypos;
    m_subframeId = 0;
}


void DemandTextureApp3D::pollKeys()
{
    const float pan = 0.04f;
    const float vpan = 0.01f;
    const float rot = 0.003f;

    float3 U, V, W;
    m_camera.UVWFrame( U, V, W );

    if( glfwGetKey( getWindow(), GLFW_KEY_A ) )
        panCamera( normalize(U) * -pan );
    if( glfwGetKey( getWindow(), GLFW_KEY_D ) )
        panCamera( normalize(U) * pan );
    if( glfwGetKey( getWindow(), GLFW_KEY_S ) )
        panCamera( normalize( float3{V.x, V.y, 0.0f} ) * -pan );
    if( glfwGetKey( getWindow(), GLFW_KEY_W ) )
        panCamera( normalize( float3{V.x, V.y, 0.0f} ) * pan );
    if( glfwGetKey( getWindow(), GLFW_KEY_Q ) )
        panCamera( float3{0.0f, 0.0f, -vpan} );
    if( glfwGetKey( getWindow(), GLFW_KEY_E ) )
        panCamera( float3{0.0f, 0.0f, vpan} );
    if( glfwGetKey( getWindow(), GLFW_KEY_J ) )
        rotateCamera( rot );
    if( glfwGetKey( getWindow(), GLFW_KEY_L ) )
        rotateCamera( -rot );
}
