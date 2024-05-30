//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

/// \file CuOmmBaking.h 
/// Primary interface of the Cuda Opacity Micromap Baking library.

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <functional>
#include <optix.h>

#if OPTIX_VERSION < 70600
#  error "CuOmmBaking requires OptiX SDK 7.6 or later"
#endif

namespace cuOmmBaking {

    /// 
    enum class Result : uint32_t
    {
        SUCCESS,                    ///< Operation succeeded.
        ERROR_CUDA,                 ///< Cuda error occurred.
        ERROR_INTERNAL,             ///< Internal baking error occurred.
        ERROR_INVALID_VALUE,        ///< Invalid argument value encountered.
        ERROR_MISALIGNED_ADDRESS,   ///< Argument buffer has misaligned address.

        MAX_NUM
    };

    /// Flags used in BakeOptions::flags.
    enum class BakeFlags : uint32_t {
        NONE                  = 0u,
        ENABLE_POST_BAKE_INFO = 1u << 1 ///< Baking will write post bake info.
    };

    // Define flag operators.
    inline constexpr BakeFlags operator|( BakeFlags x, BakeFlags y ) { return static_cast< BakeFlags > ( static_cast< uint32_t >( x ) | static_cast< uint32_t >( y ) ); }
    inline constexpr BakeFlags operator&( BakeFlags x, BakeFlags y ) { return static_cast< BakeFlags > ( static_cast< uint32_t >( x ) & static_cast< uint32_t >( y ) ); }
    inline constexpr BakeFlags operator~( BakeFlags x) { return static_cast< BakeFlags > ( ~static_cast< uint32_t >( x ) ); }

    /// This struct specifies options for Opacity Micromap baking.
    struct BakeOptions
    {
        /// Flags controlling the baking.
        /// \see BakeFlags
        BakeFlags flags = BakeFlags::NONE;

        /// Configure the Opacity Micromap format.
        /// 
        /// * OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE:
        /// Fully transparent (micro)triangles are marked as transparent.
        /// All other (micro)triangles are marked as opaque.
        /// 
        /// * OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE:
        /// Fully transparent or opaque (micro)triangles are respectively marked as transparent or opaque.
        /// All other (micro)triangles are marked as unknown-opaque.
        /// 
        /// \see OptixOpacityMicromapFormat
        OptixOpacityMicromapFormat format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;

        /// Maximum size of the output Opacity Micromap Array data in bytes.
        /// This size limit is a hard constraint, the output value BakeBuffers::outputBufferSizeInBytes will not exceed this limit.
        /// If set to zero, a default size is selected based on texture input resolutions.
        ///
        /// \see BakeBuffers::outputBufferSizeInBytes
        unsigned int maximumSizeInBytes = 0;

        /// Configure the target subdivision level.
        /// Subdivision levels are chosen such that a single micro-triangle covers approximately a 
        /// subdivisionScale * subdivisionScale texel area.
        /// 
        /// The subdivisionScale is a soft constraint. If the requested subdivisionScale would lead to 
        /// the Opacity Micromap Data exceeding the size limit, the
        /// subdivisionScale will be increased to satisfy the size limit.
        /// 
        /// If set to zero, no target subdivision level is used.
        float subdivisionScale = 0.5f;
    };

    /// Format of texture coordinates used in BakeInputDesc::texCoordFormat.
    enum class TexCoordFormat : uint32_t
    {
        UV32_FLOAT2, ///< Pair of 32-bit floats, 32-bit aligned.

        MAX_NUM
    };

    /// Format of indices used in BakeInputDesc::indexFormat, BakeInputDesc::textureIndexFormat and BakeBuffers::indexFormat.
    enum class IndexFormat : uint32_t
    {
        NONE,     ///< No indices are uses.
        I8_UINT,  ///< 8-bit unsigned integer indices. 8-bit aligned.
        I16_UINT, ///< 16-bit unsigned integer indices. 16-bit aligned.
        I32_UINT, ///< 32-bit unsigned integer indices. 32-bit aligned.

        MAX_NUM
    };

    /// Format of transform used in BakeInputDesc::transformFormat.
    enum class UVTransformFormat : uint32_t
    {
        NONE,            ///< No transform is applied to the texture coordinates.
        MATRIX_FLOAT2X3, ///< 2x3 row major affine matrix, 64-bit aligned.

        MAX_NUM
    };

    /// Values of 2-bit texels in state textures.
    /// \see StateTextureDesc
    enum class OpacityState : uint8_t
    {
        STATE_TRANSPARENT = 0,  ///< The texel is transparent.
        STATE_OPAQUE      = 1,  ///< The texel is opaque.
        STATE_UNKNOWN     = 2,  ///< The texel area opacity is neither fully transparent nor fully opaque. Point sampled opacity requires evaluation in an anyhit program.
        STATE_RESERVED    = 3
    };

    /// This struct specifies a compact opacity texture input with 2 bits per texel.
    /// Custom texture formats and procedural texures should be baked to this format for use as opacity micromap baking texture inputs.
    struct StateTextureDesc
    {
        /// The width of the texture resource in texels.
        unsigned int width;
        /// The height of the texture resource in texels.
        unsigned int height;

        /// The pitch in bits between consecutive rows of texels.
        /// The pitch must be multiple of two bits.
        /// If set to zero, states are assumed to be tightly
        /// packed with a stride of 2 bit.
        unsigned int pitchInBits;

        /// A device pointer to a buffer of rows of tightly packed 2-bit opacity states.
        /// The states in the buffer may either represent the point sampled opacities of 
        /// texels in a discrete texture (similar to the CUDA texture inputs), 
        /// or may represent the opacities of a continuous procedural pre-filtered over 
        /// texel areas. When opacity is pre-filtered the filter kernel width 
        /// should be set to zero.
        /// 
        /// \see OpacityState.
        /// \see StateTextureDesc::filterKernelWidthInTexels.
        CUdeviceptr stateBuffer;

        /// Maximum texture filtering kernel width in texture space.
        /// The baked opacity micromap data will be conservative for textures sampled using a filter kernel no larger than the specified width.
        /// For texture inputs with pre-filtered opacity a filter width of zero should be used.
        /// 
        /// \see StateTextureDesc::stateBuffer
        float filterKernelWidthInTexels;

        /// The horizontal and vertical texture address modes.
        cudaTextureAddressMode addressMode[2];
    };

    /// Mode specifying how texels in a cuda texture map to alpha values.
    enum class CudaTextureAlphaMode : uint32_t
    {
        /// The alpha mode is based on the cuda texture channel format.
        /// If   the texture has W channel: CHANNEL_W
        /// Elif the texture has Z channel: RGB_INTENSITY
        /// Elif the texture has Y channel: CHANNEL_Y
        /// Elif the texture has X channel: CHANNEL_X
        DEFAULT = 0,

        CHANNEL_X, ///< Channel X contains alpha.
        CHANNEL_Y, ///< Channel Y contains alpha.
        CHANNEL_Z, ///< Channel Z contains alpha.
        CHANNEL_W, ///< Channel W contains alpha.

        RGB_INTENSITY, ///< Alpha equals (X + Y + Z)/3

        MAX_NUM
    };

    /// This struct specifies a cuda texture input.
    struct CudaTextureDesc
    {
        /// Specifies how texel values are mapped to alpha values.
        /// \see CudaTextureAlphaMode
        CudaTextureAlphaMode alphaMode;

        /// Alpha values below the cutoff are treated as transparent.
        /// Alpha values between the transparency cutoff and the opacity cutoff are treated as unknown opacity.
        /// Note that for integer cuda textures using cudaReadModeElementType the cutoff value should be within the element type range 
        /// (i.e. for 8-bit the cutoff should be between 0 and 255)
        /// \see CudaTextureDesc::opacityCutoff
        /// \see OpacityState
        float transparencyCutoff;

        /// Alpha values equal and above the cutoff are treated as opaque.
        /// Alpha values between the transparency cutoff and the opacity cutoff are treated as unknown opacity.
        /// 
        /// Note that for integer cuda textures using cudaReadModeElementType the cutoff value should be within the element type range 
        /// (i.e. for an 8-bit alpha channel the cutoff should be between 0 and 255)
        /// 
        /// When both transparencyCutoff and opacityCutoff are left zero for cuda textures with unsigned channel format, 
        /// the opacityCutoff is set to the maximum representable alpha value for the specified cuda texture.
        /// (i.e. 1 for cudaReadModeNormalizedFloat and ( (1 << bits) - 1 ) for cudaReadModeElementType)
        /// 
        /// \see CudaTextureDesc::transparencyCutoff
        /// \see OpacityState
        float opacityCutoff;

        /// Maximum texture filtering kernel width in texture space.
        /// The baked opacity micromap data will be conservative for textures sampled using a filter kernel no larger than the specified width.
        /// When the filter width is zero, the cuda texture filter mode is used to select the appropriate filter width i.e. zero for cudaFilterModePoint and one for cudaFilterModeLinear.
        ///
        /// \see cudaTextureFilterMode
        float filterKernelWidthInTexels;

        /// Baked opacity micromap data will only be conservative with respect to the first mipmap level of mipmapped cuda textures.
        /// Baked opacity micromap data will be conservative with respect to all layers of a layered cuda textures.
        cudaTextureObject_t texObject;
    };

    /// Type of the texture input, used in TextureDesc::type.
    enum class TextureType : uint32_t
    {
        /// Texture description of type CudaTextureDesc
        CUDA,
        /// Texture description of type StateTextureDesc
        STATE,

        MAX_NUM
    };

    /// This struct specifies a texture input.
    struct TextureDesc
    {
        /// The type of the texture description.
        /// \see TextureType
        TextureType type;

        union
        {
            /// CUDA texture input.
            CudaTextureDesc cuda;

            /// State texture input.
            StateTextureDesc state;
        };
    };

    /// This struct specifies a single input for Opacity Micromap baking. The input specifies a textured geometry.
    /// An input is intended to match a corresponding build input to the OptiX Geometry Acceleration Structure build.
    /// 
    /// \see OptixBuildInputTriangleArray.
    /// \see optixAccelBuild.
    struct BakeInputDesc
    {
        /// Device pointer to an array of texture coordinates.
        /// The minimum alignment must match the natural alignment of the elements of the type specified in BakeInputDesc::texcoordFormat, i.e.,
        /// for UV32_FLOAT2 4-byte alignment.
        /// May be left zero for GetPreBakeInfo calls.
        CUdeviceptr texCoordBuffer;

        /// \see TexCoordFormat
        TexCoordFormat texCoordFormat;

        /// Stride between texture coordinates. If set to zero, texture coordinates are assumed to be tightly
        /// packed and stride is inferred from texCoordFormat.
        unsigned int texCoordStrideInBytes;

        /// Number of texture coordinates in BakeInputDesc::texCoordBuffer.
        /// Only required for triangle soups, i.e. when indexFormat is set to NONE.
        unsigned int numTexCoords;

        /// Optional device pointer to array of 16 or 32-bit int triplets, one triplet per triangle.
        /// The minimum alignment must match the natural alignment of the type as specified in BakeInputDesc::indexFormat, i.e.,
        /// for I32_UINT 4-byte and for I16_UINT a 2-byte alignment.
        /// May be left zero for GetPreBakeInfo calls.
        CUdeviceptr indexBuffer;

        /// Size of array in BakeInputDesc::indexBuffer. In calls to BakeOpacityMicromaps needs to be zero if indexBuffer is \c nullptr.
        unsigned int numIndexTriplets;

        /// \see IndexFormat
        IndexFormat indexFormat;

        /// Stride between triplets of indices. If set to zero, indices are assumed to be tightly
        /// packed and stride is inferred from BakeInputDesc::indexFormat.
        unsigned int indexTripletStrideInBytes;

        /// Optional device pointer to a 2x3 row major affine
        /// transformation matrix.
        /// The minimum alignment must match the natural alignment of the type as specified in BakeInputDesc::transformFormat, i.e.,
        /// for MATRIX_FLOAT2X3 8-byte alignment.
        /// May be left zero for GetPreBakeInfo calls.
        CUdeviceptr transform;

        /// \see UVTransformFormat
        UVTransformFormat transformFormat;

        /// The number of texture inputs.
        /// Must be at least 1.
        unsigned int numTextures;

        /// An array of texture inputs.
        /// BakeOpacityMicromaps compares texture descriptors by value across bake inputs to detect and 
        /// re-use duplicate Opacity Micromaps (\see BakeOpacityMicromaps).
        const TextureDesc* textures;

        /// Optional device pointer to an array of texture indices, one per triangle.
        /// Every entry must be in range [0,numTextures).
        /// May be left zero for GetPreBakeInfo calls.
        CUdeviceptr textureIndexBuffer;

        /// \see IndexFormat
        /// Must be NONE when BakeInputDesc::numTextures equals 1.
        /// Must not be NONE when BakeInputDesc::numTextures is larger than 1.
        IndexFormat textureIndexFormat;

        /// Stride between the indices. If set to zero, the indices are assumed to be tightly
        /// packed and the stride matches the size of the type specified in BakeInputDesc::textureIndexFormat.
        unsigned int textureIndexStrideInBytes;
    };

    /// This struct specifies the actual sizes of data output by baking.
    /// The output buffers may be compacted to these sizes to reclaim unused memory.
    /// These sizes are not a required input for the OptiX Opacity Micromap Array build,
    /// but may be used to compact offline baked Opacity Micromap data before storage.
    struct PostBakeInfo
    {
        /// Number of used elements in BakeBuffers::perMicromapDescBuffer.
        /// Matches the sum of counts over all histogram entries in BakeBuffers::micromapHistogramEntriesBuffer.
        unsigned int numMicromapDescs;

        /// Used bytes in BakeBuffers::outputBuffer.
        /// This value is at most BakeBuffers::outputBufferSizeInBytes.
        /// The output buffer may be compacted to a buffer if this size.
        unsigned int compactedSizeInBytes;
    };

    /// Required alignment in bytes for output buffers.
    /// 
    /// \see BakeInputBuffers
    /// \see BakeBuffers
    enum BufferAlignmentInBytes
    {
        /// The minimum required memory alignment in bytes for the device output buffer BakeInputBuffers::micromapUsageCountsBuffer.
        MICROMAP_USAGE_COUNTS      = 4,

        /// The minimum required memory alignment in bytes for the device output buffer BakeBuffers::outputBuffer.
        MICROMAP_OUTPUT            = 8,

        /// The minimum required memory alignment in bytes for the device output buffer BakeBuffers::micromapHistogramEntriesBuffer.
        MICROMAP_HISTOGRAM_ENTRIES = 4,

        /// The minimum required memory alignment in bytes for the device output buffer BakeBuffers::perMicromapDescBuffer.
        MICROMAP_DESC              = 8,

        /// The minimum required memory alignment in bytes for the device output buffer BakeBuffers::postBakeInfoBuffer.
        POST_BAKE_INFO             = 8,

        /// The minimum required memory alignment in bytes for the device output buffer BakeBuffers::tempBuffer.
        TEMP                       = 1
    };

    /// This struct specifies the baking output buffers for a single bake input.
    /// The minimum required sizes of the buffers are initialized by calling #GetPreBakeInfo.
    /// These output buffers are intended as inputs to the OptiX Geometry Acceleration Structure build. 
    /// 
    /// \see OptixBuildInputOpacityMicromap.
    /// \see OptixBuildInputTriangleArray::opacityMicromap.
    /// \see optixAccelBuild.
    struct BakeInputBuffers
    {
        /// Device pointer to a int16 or int32 output buffer specifying which opacity micromap index to use for each triangle in a bake input.
        /// The type of this buffer is specified by BakeBuffers::indexFormat and is quired using #GetPreBakeInfo.
        /// This buffer is intended as input to the OptiX Geometry Acceleration Structure build.
        /// The minimum alignment must match the natural alignment of the type as specified in BakeBuffers::indexFormat, i.e.,
        /// for I16_UINT 2-byte alignment and I32_UINT 4-byte alignment.
        /// 
        /// \see OptixBuildInputOpacityMicromap::indexBuffer
        CUdeviceptr indexBuffer;
        /// Size of BakeInputBuffers::indexBuffer
        size_t      indexBufferSizeInBytes;

        /// Device pointer to an output buffer of OptixOpacityMicromapUsageCount.
        /// This buffer is generally downloaded to the host after baking for use as input to the OptiX Geometry Acceleration Structure build
        /// 
        /// \see OptixBuildInputOpacityMicromap::micromapUsageCounts, OptixOpacityMicromapUsageCount.
        /// \see BufferAlignmentInBytes::MICROMAP_USAGE_COUNTS
        CUdeviceptr micromapUsageCountsBuffer;
        /// Number of elements in BakeInputBuffers::micromapUsageCountsBuffer.
        size_t      numMicromapUsageCounts;
    };

    /// This struct specifies the baking output buffers.
    /// The format and minimum required sizes of the buffers are initialized by calling #GetPreBakeInfo.
    struct BakeBuffers
    {
        /// Format of indices in BakeInputBuffers::indexBuffer.
        /// 
        /// \see OptixBuildInputOpacityMicromap::indexSizeInBytes
        IndexFormat indexFormat;

        /// Device pointer to an Opacity Micromap Array opacity output buffer.
        /// This buffer is intended as input for the OptiX Opacity Micromap Array build.
        ///
        /// \see OptixOpacityMicromapArrayBuildInput::inputBuffer
        /// \see BufferAlignmentInBytes::MICROMAP_OUTPUT
        CUdeviceptr outputBuffer;
        /// Size of BakeBuffers::outputBuffer
        /// The actual size of the baked Opacity Micromap Array Data may be smaller.
        /// The final size is obtained from the post bake info.
        /// 
        /// \see PostBakeInfo::compactedSizeInBytes
        size_t      outputBufferSizeInBytes;

        /// Device pointer to an output buffer of OptixOpacityMicromapDesc.
        /// This buffer is intended as input for the OptiX Opacity Micromap Array build.
        /// 
        /// \see OptixOpacityMicromapArrayBuildInput::perMicromapDescBuffer
        /// \see OptixOpacityMicromapDesc
        /// \see BufferAlignmentInBytes::MICROMAP_DESC
        CUdeviceptr perMicromapDescBuffer;
        /// Number of elements in BakeBuffers::perMicromapDescBuffer
        /// The actual descriptor count in the baked Opacity Micromap Array may be smaller.
        /// The final count is obtained from the post bake info.
        /// \see PostBakeInfo::numMicromapDescs
        size_t      numMicromapDescs;

        /// Device pointer to an output buffer of OptixOpacityMicromapHistogramEntry.
        /// This buffer is generally downloaded to the host after baking for use as input to OptiX Opacity Micromap Array build .
        /// 
        /// \see OptixOpacityMicromapArrayBuildInput::micromapHistogramEntries
        /// \see OptixOpacityMicromapHistogramEntry
        /// \see BufferAlignmentInBytes::MICROMAP_HISTOGRAM_ENTRIES
        CUdeviceptr micromapHistogramEntriesBuffer;
        /// Number of elements in BakeBuffers::micromapHistogramEntriesBuffer.
        size_t      numMicromapHistogramEntries;

        /// Device pointer to an output buffer of type PostBakeInfo. 
        /// The post bake info is not required to build an OptiX Opacity Micromap Array.
        /// This output is optional and enabled in the options using the baking flag BakeFlags::ENABLE_POST_BAKE_INFO
        /// 
        /// \see PostBakeInfo
        /// \see BufferAlignmentInBytes::POST_BAKE_INFO
        /// \see BakeFlags::ENABLE_POST_BAKE_INFO
        /// \see BakeOptions::flags
        CUdeviceptr postBakeInfoBuffer;
        /// Size of BakeBuffers::postBakeInfoBuffer
        size_t      postBakeInfoBufferSizeInBytes;

        /// Device pointer to a temporary memory buffer used during the baking.
        /// \see BufferAlignmentInBytes::TEMP
        CUdeviceptr tempBuffer;
        /// Size of BakeBuffers::tempBuffer
        size_t      tempBufferSizeInBytes;
    };

    /// Fills in pre bake information for baking.
    ///
    /// To prepare for baking, baking information is queried by passing a set of bake inputs and parameters to GetPreBakeInfo.
    /// GetPreBakeInfo initializes the output format and the required sizes for 
    /// all baking output buffers (\see BakeInputBuffers, \see BakeBuffers).
    /// 
    /// This function is thread-safe. 
    /// This function does NOT execute any device tasks, does NOT access input/output device memory and does NOT synchronize with the device.
    /// Device buffers specified in BakeInputDesc may be left zero for this call.
    ///  
    /// \param[in]  options                 Baking options.
    /// \param[in]  numInputs               Number of elements in inputs (must be at least 1).
    /// \param[in]  inputs                  An array of baking inputs.
    /// \param[out] outInputBuffers         An array of output structs initialized by the call, specifying per bake input the required sizes of baking output buffers. 
    /// \param[out] outBuffers              An output struct initialized by the call, specifies the required sizes and format of baking output buffers.
    Result GetPreBakeInfo(
        const BakeOptions*     options,
        unsigned               numInputs,
        const BakeInputDesc*   inputs,
        BakeInputBuffers*      outInputBuffers,
        BakeBuffers*           outBuffers );

    /// Execute Opacity Micromap Baking tasks on the device.
    ///
    /// This function executes the baking of opacity micromaps for textured geometry.
    /// This function writes the data buffers needed to build an OptiX Opacity Micromap Array.
    /// Furthermore, this function writes the data buffers needed to build an OptiX Geometry Acceleration Structure using the 
    /// OptiX Opacity Micromap Array.
    /// This function does not allocate any device memory. The user has to provide the output buffer device memory.
    /// The required format and sizes of all output buffers are obtained by calling GetPreBakeInfo with the same option and bake input parameters.
    /// 
    /// This function is thread-safe. 
    /// This function does NOT synchronize with the device, all device tasks is executed asynchronously in the provided stream.
    ///  
    /// \param[in] options            Baking options.
    /// \param[in] numInputs          Number of elements in inputs (must be at least 1).
    /// \param[in] inputs             An array of BakeInputDesc objects.
    /// \param[in] inputBuffers       An array of structs, specifying device output buffers per bake input.
    /// \param[in] buffers            A struct specifying the device output buffers.
    /// \param[in] stream             Optional CUDA stream to launch kernels within.
    Result BakeOpacityMicromaps(
        const BakeOptions*       options,
        unsigned                 numInputs,
        const BakeInputDesc*     inputs,
        const BakeInputBuffers*  inputBuffers,
        const BakeBuffers*       buffers,
        cudaStream_t             stream = 0 );

}  // namespace cuOmmBaking
