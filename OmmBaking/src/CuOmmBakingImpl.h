// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/CuOmmBaking/CuOmmBaking.h>

#include <cuda_runtime.h>
#include <optix.h>

struct TextureData
{
    inline __device__ __host__ int getColPitch() const { return height; }

    inline __device__ __host__ int getRowPitch() const { return 1; }

    // texel resolution
    uint32_t width;
    uint32_t height;

    /// filter kernel for conervative opacity evaluation.
    float filterKernelRadiusInTexels;

    /// horizontal and vertical address modes.
    cudaTextureAddressMode addressMode[2];

    /// (num-transparent,num-opaque) state sum table
    /// table is in column major order for efficient construction.
    /// is null for custom texture inputs
    const uint2* sumTable;

    ///  identifier used for duplicate detection
    uint32_t id;
};

struct TextureInput
{
    // Texture data description
    TextureData data;

    /// quantization resolution used for matching of near-identical uvs.
    float2 quantizationFrequency;
    float2 quantizationPeriod;

    /// periodicity of the texture data in quantized units. 
    /// used for unwrapping before de-duplication.
    float2 quantizedPeriod;
    float2 quantizedFrequency;
};

struct BakeInput
{
    void* outAssignments;

    const TextureInput* inTextures;

    cuOmmBaking::BakeInputDesc desc;
};

struct TriangleID
{
    uint32_t triangleIndex : 32; // triangle index within the input.
    uint32_t inputIndex    : 28; // bake input index.
    uint32_t uniform       : 1;  // set if the triangle has a uniform opacity state.
    uint32_t state         : 2;  // uniform opacity state.
    uint32_t reserved      : 1;  // unused.
};

struct SetupBakeInputParams
{
    // number of input triangles.
    uint32_t numTriangles;
    // index of the current input.
    uint32_t inputIdx;

    // ids for all input triangles.
    TriangleID* outTriangleIDs;
    // hash keys for all input triangles.
    uint32_t* outHashKeys;

    // texture inputs for this bake input
    const TextureInput* textures;

    // the user input descriptor
    cuOmmBaking::BakeInputDesc input;

    // opacity micromap output format
    OptixOpacityMicromapFormat format;
};

struct MarkFirstOmmOccuranceParams
{
    uint32_t numTriangles;

    // list of triangle ids, sorted by hask key.
    const TriangleID* inTriangleIDs;
    // corresponding sorted hash keys.
    const uint32_t* inHashKeys;
    // bake inputs.
    const BakeInput* inBakeInputs;

    // texture inputs for this bake input
    const TextureInput* textures;

    // the first occurance of each unique opacity micromap is marked with a one.
    uint32_t* outMarkers;
};

struct GenerateAssignmentParams
{
    uint32_t numTriangles;

    // upper bound on the number of omms.
    uint32_t maxOmms;

    // final number of omms in the array
    uint32_t* outNumOmms;

    // Format of omm indices.
    cuOmmBaking::IndexFormat indexFormat;

    // list of triangle ids, sorted by hask key.
    const TriangleID* inTriangleIDs;
    // the flat omm assignment over all triangles plus one.
    const uint32_t* inAssignment;

    // bake inputs.
    const BakeInput* inBakeInputs;
    // per omm triangle ID of one of the triangles in the dupplicate group
    TriangleID* outOmmTriangleId;

    // area in texels per opacticy micromap.
    float* outOmmArea;
};

struct GenerateLayoutParams
{
    // per opacticy micromaps area in texels.
    const float* inOmmArea;
    // total area summed over all opacticy micromaps.
    const float* inSumArea;
    // total number of opacticy micromaps.
    const uint32_t* inNumOmms;
    // input buffer of opacticy micromap weights.
    // output buffer of opacticy micromap descriptors.
    OptixOpacityMicromapDesc* ioDescs;
    // upper limit on the size of the total raw opacticy micromap data.
    uint32_t maxOmmArraySizeInBytes;
    // final size in bytes. must be zero before launch.
    uint32_t* ioSizeInBytes;
    // histogram of omms per subdiv level. must be zero before launch.
    OptixOpacityMicromapHistogramEntry* ioHistogram;
    // target subdivision level in micro-triangles per texel.
    float microTrianglesPerTexel;
    // opacity micromap target format.
    OptixOpacityMicromapFormat format;
};

struct GenerateInputHistogramParams
{
    uint32_t numTriangles;

    // Format of omm indices.
    cuOmmBaking::IndexFormat indexFormat;
    // the flat omm assignment over all triangles.
    const void* inAssignment;
    // descriptors for each opacticy micromap.
    const OptixOpacityMicromapDesc* inDescs;
    // per bake input histograms of omms per subdiv level. must be zero before launch.
    OptixOpacityMicromapUsageCount* ioHistogram;
};

struct EvaluateOmmOpacityParams
{
    // data size in bytes.
    const uint32_t* inSizeInBytes;

    // total number of opacticy micromaps.
    const uint32_t* inNumOmms;

    // descriptors for each opacticy micromap.
    const OptixOpacityMicromapDesc* inDescs;

    // raw micro triangle opacity data.
    // must be zeroed before launch.
    void* ioData;

    // per omm triangle ID of one of the triangles in the dupplicate group.
    const TriangleID* inTriangleIdPerOmm;

    // bake input descriptors.
    const BakeInput* inBakeInputs;

    // total size of the output buffer. for validation only.
    uint32_t dataSizeInBytes;

    // opacity micromap target format.
    OptixOpacityMicromapFormat format;
};

// setup all a single bake input.
cudaError_t launchSetupBakeInput( SetupBakeInputParams params, cudaStream_t stream );

// scan the sorted triangle list and mark the start of duplicate groups.
cudaError_t launchMarkFirstOmmOccurance( MarkFirstOmmOccuranceParams params, cudaStream_t stream );

// generate omm assignments for all triangles in all bake inputs.
cudaError_t launchGenerateAssignment( GenerateAssignmentParams params, cudaStream_t stream );

// generate the omm layout, dynamically assigning subdivision levels.
cudaError_t launchGenerateLayout( GenerateLayoutParams params, unsigned int numThreads, cudaStream_t stream );

// generate omm descriptor byte offsets from subdivision levels.
cudaError_t launchGenerateStartOffsets(
    void*                           temp,
    size_t&                         tempSizeInBytes,
    const OptixOpacityMicromapDesc* inDesc,
    OptixOpacityMicromapDesc*       outDesc,
    unsigned int                    numItems,
    OptixOpacityMicromapFormat      format,
    cudaStream_t                    stream );

// generate the subdivision level usage histograms for all bake inputs.
cudaError_t launchGenerateInputHistogram( GenerateInputHistogramParams params, cudaStream_t stream );

// evaluate the opacity of all micro triangles in all opacity micro maps.
cudaError_t launchEvaluateOmmOpacity( EvaluateOmmOpacityParams params, unsigned int numThreads, cudaStream_t stream );

// Functor encapsulating cuda cub Reduction.
// The template implementations are exlicitly instanciated in OmmBakingImpl.cu
template <
    typename InputIteratorT,
    typename OutputIteratorT,
    typename T>
struct ReduceRoundUp
{
    cudaError_t operator()(
        void*           d_temp_storage,     ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&         temp_storage_bytes, ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in,               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT d_out,              ///< [out] Pointer to the output aggregate
        int             num_items,          ///< [in] Total number of input items (i.e., length of \p d_in)
        cudaStream_t    stream = 0          ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
      ) const; 
};

// Functor encapsulating cuda cub InclusiveSum.
// The template implementations are exlicitly instanciated in OmmBakingImpl.cu
template <typename InputIteratorT, typename OutputIteratorT>
struct InclusiveSum
{
    cudaError_t operator()(
        void*           d_temp_storage,      ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&         temp_storage_bytes,  ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in,                ///< [in] Pointer to the input sequence of data items
        OutputIteratorT d_out,               ///< [out] Pointer to the output sequence of data items
        int             num_items,           ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t    stream = 0           ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
      ) const;
};

// Functor encapsulating cuda cub SortPairs.
// The template implementations are exlicitly instanciated in OmmBakingImpl.cu
template <typename KeyT, typename ValueT>
struct SortPairs
{
    cudaError_t operator()( 
        void*         d_temp_storage,                ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&       temp_storage_bytes,            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        const KeyT*   d_keys_in,                     ///< [in] Pointer to the input data of key data to sort
        KeyT*         d_keys_out,                    ///< [out] Pointer to the sorted output sequence of key data
        const ValueT* d_values_in,                   ///< [in] Pointer to the corresponding input sequence of associated value items
        ValueT*       d_values_out,                  ///< [out] Pointer to the correspondingly-reordered output sequence of associated value items
        int           num_items,                     ///< [in] Number of items to sort
        int           begin_bit = 0,                 ///< [in] <b>[optional]</b> The least-significant bit index (inclusive)  needed for key comparison
        int           end_bit = sizeof( KeyT ) * 8,  ///< [in] <b>[optional]</b> The most-significant bit index (exclusive) needed for key comparison (e.g., sizeof(unsigned int) * 8)
        cudaStream_t  stream = 0                     ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
      ) const;
};
