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

#include "Util/BufferLayout.h"
#include "Util/Exception.h"

#include "CuOmmBakingImpl.h"
#include "Texture.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

    std::ostream& operator<<( std::ostream& os, const uint2 v  )
    {
        os << "[" << v.x << "," << v.y << "]";
        return os;
    }

    std::ostream& operator<<( std::ostream& os, const OptixOpacityMicromapDesc& desc )
    {
        os << "byteOffset=" << desc.byteOffset << ", format=" << desc.format << ", subdivisionLevel=" << desc.subdivisionLevel;
        return os;
    }

    // Use compiler annotations to declare the argument to be a printf-style format string
    // to get format specifier checking against actual arguments.
#if defined( __GNUC__ )
#define RTC_PRINTF_FORMAT( n_, m_ ) __attribute__( ( format( printf, n_, m_ ) ) )
#define RTC_PRINTF_ARGUMENT
#elif defined( _MSC_VER )
#define RTC_PRINTF_FORMAT( n_, m_ )
#define RTC_PRINTF_ARGUMENT _Printf_format_string_
#else /* __GNUC__ || _MSC_VER */
#error "Unknown compiler.  You need to figure out how to declare format string arguments."
#endif
// Create a string with printf-like arguments
    inline RTC_PRINTF_FORMAT( 1, 2 ) std::string stringf( RTC_PRINTF_ARGUMENT const char* fmt, ... )
    {
        va_list args;
        va_start( args, fmt );
#ifdef WIN32
        int size = _vscprintf( fmt, args );
#else
        int size = vsnprintf( nullptr, 0, fmt, args );
#endif
        va_end( args );

        std::string ret;
        if( size > 0 )
        {
            ret.resize( size );
            va_start( args, fmt );
            vsnprintf( ( char* )ret.data(), size + 1, fmt, args );
            va_end( args );
        }
        return ret;
    }

    template<typename T>
    constexpr bool isAligned( CUdeviceptr ptr )
    {
        return ( ptr & ( highestPowerOf2( sizeof( T ) ) - 1 ) ) == 0;
    }
}

namespace cuOmmBaking {

// Unclear what the alignment requirements of cub tempoary buffers are, so stick to something conservative
#define CUB_TEMP_BUFFER_ALIGNMENT_IN_BYTES 128

// The intermediate horizontally summed values are 16 bit
#define SAT_TEMP_BUFFER_ALIGNMENT_IN_BYTES sizeof(ushort2)

template <typename T>
void CudaMemcpyAsync( BufferLayout<T>& out, const std::vector<T>& in, cudaStream_t stream )
{
    OMM_CUDA_CHECK( cudaMemcpyAsync( out.access(), in.data(), out.getNumBytes(), cudaMemcpyHostToDevice, stream ) );
}


template <typename T>
void CudaMemsetAsync( BufferLayout<T>& out, cudaStream_t stream )
{
    OMM_CUDA_CHECK( cudaMemsetAsync( out.access(), 0, out.getNumBytes(), stream ) );
}

uint32_t getNumTriangles( const BakeInputDesc& input )
{
    return ( input.indexFormat != IndexFormat::NONE ? input.numIndexTriplets : ( input.numTexCoords / 3 ) );
}

template <typename T>
void dump( const char* name, BufferLayout<T>& buffer )
{
    std::vector<T>  h( buffer.getNumElems() );
    OMM_CUDA_CHECK( cudaMemcpyAsync( h.data(), buffer.access(), buffer.getNumBytes(), cudaMemcpyDeviceToHost ) );
    std::cout << name << std::endl;
    for( size_t i = 0; i < h.size(); i++ )
    {
        std::cout << "\t" << i << ": " << h[i] << std::endl;
    }
}

bool operator<( const TextureDesc& a, const TextureDesc& b )
{
    #define COMPARE_TEXTURE_DESC_MEMBER(x) {    \
        if( a.x < b.x )                         \
            return true;                        \
        if( a.x > b.x )                         \
            return false;                       \
    }

    COMPARE_TEXTURE_DESC_MEMBER( type );

    if( a.type == TextureType::CUDA )
    {
        // Compare the texture object first as early out. distinct textures will usually disagree here.
        COMPARE_TEXTURE_DESC_MEMBER( cuda.texObject );
        COMPARE_TEXTURE_DESC_MEMBER( cuda.transparencyCutoff );
        COMPARE_TEXTURE_DESC_MEMBER( cuda.opacityCutoff );
        COMPARE_TEXTURE_DESC_MEMBER( cuda.alphaMode );

        // The SAT could be re-used when an otherwise identical texture desc 
        // uses a different filter width.
        COMPARE_TEXTURE_DESC_MEMBER( cuda.filterKernelWidthInTexels );
    }
    else // TextureType::STATE
    {
        // Compare the state buffer first as early out. distinct textures will usually disagree here.
        COMPARE_TEXTURE_DESC_MEMBER( state.stateBuffer );
        COMPARE_TEXTURE_DESC_MEMBER( state.width );
        COMPARE_TEXTURE_DESC_MEMBER( state.height );
        COMPARE_TEXTURE_DESC_MEMBER( state.pitchInBits );

        // The SAT could be re-used when an otherwise identical texture desc 
        // uses a different filter width or addressing modes.
        COMPARE_TEXTURE_DESC_MEMBER( state.filterKernelWidthInTexels );
        COMPARE_TEXTURE_DESC_MEMBER( state.addressMode[0] );
        COMPARE_TEXTURE_DESC_MEMBER( state.addressMode[1] );
    }

#undef COMPARE_TEXTURE_DESC_MEMBER

    return false;
}


class Baker
{
public:

    // Base class for internal device texture object
    struct TextureBase
    {
    public:

        TextureBase( const TextureBase& ) = delete;
        TextureBase& operator=( const TextureBase& ) = delete;

        TextureBase( uint32_t id, uint32_t width, uint32_t height )
            : m_id( id )
        {
            m_satBuf.setNumBytes( width * height * sizeof( uint2 ) );
        }

        virtual ~TextureBase() {};

        virtual cudaError_t build( void* temp, size_t& tempStorageInBytes, cudaStream_t stream ) = 0;

        virtual TextureData get( const cuOmmBaking::TextureDesc& desc )
        {
            TextureData textureInput = {};
            textureInput.id = m_id;
            textureInput.sumTable = m_satBuf.access();
            return textureInput;
        }

        template<typename T>
        void aggregateInto( BufferLayout<T>& parent )
        {
            parent.aggregate( m_satBuf );
        }

    protected:

        BufferLayout<uint2> m_satBuf;
        uint32_t            m_id = {};
    };

    typedef std::map<TextureDesc, std::shared_ptr<TextureBase>> TextureMap;

    struct StateTexture : public TextureBase
    {
    public:
        StateTexture(
            uint32_t           id,
            const uint8_t*     input,
            StateTextureConfig config )
            : TextureBase( id, config.width, config.height )
            , m_input( input )
            , m_config( config )
        {}

        cudaError_t build( void* temp, size_t& tempStorageInBytes, cudaStream_t stream )
        {
            return ::launchSummedAreaTable( temp, tempStorageInBytes, m_config, m_input, m_satBuf.isMaterialized() ? m_satBuf.access() : 0, stream );
        }

        virtual TextureData get( const cuOmmBaking::TextureDesc& desc )
        {
            TextureData input = TextureBase::get( desc );
            input.width = m_config.width;
            input.height = m_config.height;
            input.addressMode[0] = desc.state.addressMode[0];
            input.addressMode[1] = desc.state.addressMode[1];
            input.filterKernelRadiusInTexels = 0.5f * desc.state.filterKernelWidthInTexels;
            return input;
        }
    private:
        const uint8_t*     m_input  = {};
        StateTextureConfig m_config = {};
    };

    struct CudaTexture : public TextureBase
    {
    public:
        CudaTexture(
            uint32_t               id,
            cudaTextureObject_t    texture,
            cudaTextureAddressMode addressMode[2],
            float                  filterKernelWidthInTexels,
            CudaTextureConfig      config )
            : TextureBase( id, config.width, config.height )
            , m_texture( texture )
            , m_filterKernelWidthInTexels( filterKernelWidthInTexels )
            , m_config( config )
        {
            m_addressMode[0] = addressMode[0];
            m_addressMode[1] = addressMode[1];
        }

        cudaError_t build( void* temp, size_t & tempStorageInBytes, cudaStream_t stream )
        {
            return ::launchSummedAreaTable( temp, tempStorageInBytes, m_config, m_texture, m_satBuf.isMaterialized() ? m_satBuf.access() : 0, stream );
        }

        virtual TextureData get( const cuOmmBaking::TextureDesc& desc )
        {
            TextureData input = TextureBase::get( desc );
            input.width = m_config.width;
            input.height = m_config.height;
            input.addressMode[0] = m_addressMode[0];
            input.addressMode[1] = m_addressMode[1];
            input.filterKernelRadiusInTexels = 0.5f * m_filterKernelWidthInTexels;
            return input;
        }
    private:
        cudaTextureObject_t    m_texture        = {};
        cudaTextureAddressMode m_addressMode[2] = {};
        CudaTextureConfig      m_config         = {};

        float m_filterKernelWidthInTexels = 0.f;
    };

    Baker(
        const BakeOptions*      options,
        unsigned                numInputs,
        const BakeInputDesc*    inputs,
        const BakeInputBuffers* inputBuffers,
        const BakeBuffers*      buffers )
    {
        if( options == 0 )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for options. Must not be zero." ) );

        if( numInputs == 0 )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for numInputs. Must not be zero." ) );

        if( inputs == 0 )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs. Must not be zero." ) );

        bool isPreBake = ( buffers == 0 );

        validate( *options );

        for( unsigned int i = 0; i < numInputs; ++i )
            validate( inputs[i], i, isPreBake );

        Baker::m_options = *options;
        Baker::m_inputs = std::vector<BakeInputDesc>( inputs, inputs + numInputs );

        uint64_t numTexels = 0;
        uint32_t numTextureReferences = 0;
        uint32_t numUniqueTextures = 0;
        size_t satTempStorageInBytes = 0;
        for( size_t inputIdx = 0; inputIdx < Baker::m_inputs.size(); ++inputIdx )
        {
            auto& input = Baker::m_inputs[inputIdx];

            if( input.indexTripletStrideInBytes == 0 )
            {
                switch( input.indexFormat )
                {
                case IndexFormat::I8_UINT:
                    input.indexTripletStrideInBytes = sizeof( uchar3 );
                    break;
                case IndexFormat::I16_UINT:
                    input.indexTripletStrideInBytes = sizeof( ushort3 );
                    break;
                case IndexFormat::I32_UINT:
                    input.indexTripletStrideInBytes = sizeof( uint3 );
                    break;
                default:
                    break;
                };
            }

            if( input.texCoordStrideInBytes == 0 )
            {
                switch( input.texCoordFormat )
                {
                case TexCoordFormat::UV32_FLOAT2:
                    input.texCoordStrideInBytes = sizeof( float2 );
                    break;
                default:
                    break;
                };
            }

            if( input.textureIndexStrideInBytes == 0 )
            {
                switch( input.textureIndexFormat )
                {
                case IndexFormat::I8_UINT:
                    input.textureIndexStrideInBytes = sizeof( uint8_t );
                    break;
                case IndexFormat::I16_UINT:
                    input.textureIndexStrideInBytes = sizeof( uint16_t );
                    break;
                case IndexFormat::I32_UINT:
                    input.textureIndexStrideInBytes = sizeof( uint32_t );
                    break;
                default:
                    break;
                };
            }

            numTextureReferences += input.numTextures;
            for( size_t i = 0; i < input.numTextures; i++ )
            {
                const TextureDesc& key = input.textures[i];

                if( m_textureMap.find( key ) == m_textureMap.end() )
                {
                    // Exceding this may lead to overflows in the 32 bit summed area table.
                    const uint32_t maxExtent = ( 1u << 15 ) - 1;

                    std::shared_ptr<TextureBase> texture;

                    size_t width  = 0;
                    size_t height = 0;
                    switch( input.textures[i].type )
                    {
                    case TextureType::CUDA:
                    {
                        cudaChannelFormatDesc chanDesc = {};
                        cudaResourceDesc      resDesc = {};
                        cudaExtent            extent = {};

                        cudaTextureObject_t texObject = input.textures[i].cuda.texObject;
                        OMM_CUDA_CHECK( cudaGetTextureObjectResourceDesc( &resDesc, texObject ) );

                        switch( resDesc.resType )
                        {
                        case cudaResourceTypeArray: {
                            OMM_CUDA_CHECK( cudaGetChannelDesc( &chanDesc, resDesc.res.array.array ) );
                            OMM_CUDA_CHECK( cudaArrayGetInfo( 0, &extent, 0, resDesc.res.array.array ) );
                        } break;
                        case cudaResourceTypeMipmappedArray: {
                            cudaArray_t d_topLevelArray;
                            OMM_CUDA_CHECK( cudaGetMipmappedArrayLevel( &d_topLevelArray, resDesc.res.mipmap.mipmap, 0 ) );
                            OMM_CUDA_CHECK( cudaGetChannelDesc( &chanDesc, d_topLevelArray ) );
                            OMM_CUDA_CHECK( cudaArrayGetInfo( 0, &extent, 0, d_topLevelArray ) );
                        } break;
                        case cudaResourceTypeLinear: {
                            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid texture type for inputs[%zu].textures[%zu].cuda.texObject. Type cudaResourceTypeLinear is not supported.", inputIdx, i ) );
                        } break;
                        case cudaResourceTypePitch2D: {
                            extent.width = resDesc.res.pitch2D.width;
                            extent.height = resDesc.res.pitch2D.height;
                            chanDesc = resDesc.res.pitch2D.desc;
                        } break;
                        default: {
                            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid texture type for inputs[%zu].textures[%zu].cuda.texObject. Type %u is not supported.", inputIdx, i, resDesc.resType ) );
                        }
                        };

                        CudaTextureAlphaMode alphaMode = input.textures[i].cuda.alphaMode;
                        switch( alphaMode )
                        {
                        case CudaTextureAlphaMode::DEFAULT:
                            if( chanDesc.w != 0 )
                                alphaMode = CudaTextureAlphaMode::CHANNEL_W;
                            else if( chanDesc.x != 0 && chanDesc.y != 0 && chanDesc.z != 0 )
                                alphaMode = CudaTextureAlphaMode::RGB_INTENSITY;
                            else if( chanDesc.y != 0 && chanDesc.z == 0 )
                                alphaMode = CudaTextureAlphaMode::CHANNEL_Y;
                            else if( chanDesc.x != 0 && chanDesc.z == 0 )
                                alphaMode = CudaTextureAlphaMode::CHANNEL_X;
                            else
                                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid mode `DEFAULT` for inputs[%zu].textures[%zu].cuda.alphaMode. Texture has unsupported channel format.", inputIdx, i ) );
                            break;
                        case CudaTextureAlphaMode::CHANNEL_X:
                            if( chanDesc.x == 0 )
                                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid mode `CHANNEL_X` for inputs[%zu].textures[%zu].cuda.alphaMode. Texture has no X channel.", inputIdx, i ) );
                            break;
                        case CudaTextureAlphaMode::CHANNEL_Y:
                            if( chanDesc.y == 0 )
                                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid mode `CHANNEL_Y` for inputs[%zu].textures[%zu].cuda.alphaMode. Texture has no Y channel.", inputIdx, i ) );
                            break;
                        case CudaTextureAlphaMode::CHANNEL_Z:
                            if( chanDesc.z == 0 )
                                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid mode `CHANNEL_Z` for inputs[%zu].textures[%zu].cuda.alphaMode. Texture has no Z channel.", inputIdx, i ) );
                            break;
                        case CudaTextureAlphaMode::CHANNEL_W:
                            if( chanDesc.w == 0 )
                                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid mode `CHANNEL_W` for inputs[%zu].textures[%zu].cuda.alphaMode. Texture has no W channel.", inputIdx, i ) );
                            break;
                        case CudaTextureAlphaMode::RGB_INTENSITY:
                            if( chanDesc.x == 0 || chanDesc.y == 0 || chanDesc.z == 0 )
                                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid mode `RGB_INTENSITY` for inputs[%zu].textures[%zu].cuda.alphaMode. Texture has no X,Y and Z channels.", inputIdx, i ) );
                            break;
                        default:
                            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%zu].textures[%zu].cuda.alphaMode.", (uint32_t) input.textures[i].cuda.alphaMode, inputIdx, i ) );
                            break;
                        }

                        cudaTextureDesc texDesc;
                        OMM_CUDA_CHECK( cudaGetTextureObjectTextureDesc( &texDesc, texObject ) );

                        float filterKernelWidthInTexels = input.textures[i].cuda.filterKernelWidthInTexels;

                        if( filterKernelWidthInTexels < 0.f || std::isnan( filterKernelWidthInTexels ) )
                            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %f for inputs[%zu].textures[%zu].cuda.filterKernelWidthInTexels. Must be real positive value.", filterKernelWidthInTexels, inputIdx, i ) );

                        // By default, take the filter width from the cuda filter mode
                        if( filterKernelWidthInTexels == 0.f )
                        {
                            if( texDesc.filterMode == cudaFilterModeLinear )
                                filterKernelWidthInTexels = 1.f;
                        }

                        if( extent.width > maxExtent )
                            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid width of %zu for inputs[%zu].textures[%zu].cuda.texObject. Width must not exceed %u.", extent.width, inputIdx, i, maxExtent ) );

                        if( extent.height > maxExtent )
                            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid height of %zu for inputs[%zu].textures[%zu].cuda.texObject. Height must not exceed %u.", extent.height, inputIdx, i, maxExtent ) );

                        float transparencyCutoff = input.textures[i].cuda.transparencyCutoff;
                        float opacityCutoff = input.textures[i].cuda.opacityCutoff;

                        // Set reasonable default opacity cutoff when possible
                        if( chanDesc.f == cudaChannelFormatKindUnsigned && transparencyCutoff == 0.f && opacityCutoff == 0.f )
                        {
                            if( texDesc.readMode == cudaReadModeNormalizedFloat )
                            {
                                opacityCutoff = 1.f;
                            }
                            else // cudaReadModeElementType
                            {
                                switch( alphaMode )
                                {
                                case CudaTextureAlphaMode::CHANNEL_X:
                                    opacityCutoff = ( float )( ( 1u << chanDesc.x ) - 1 );
                                    break;
                                case CudaTextureAlphaMode::CHANNEL_Y:
                                    opacityCutoff = ( float )( ( 1u << chanDesc.y ) - 1 );
                                    break;
                                case CudaTextureAlphaMode::CHANNEL_Z:
                                    opacityCutoff = ( float )( ( 1u << chanDesc.z ) - 1 );
                                    break;
                                case CudaTextureAlphaMode::CHANNEL_W:
                                    opacityCutoff = ( float )( ( 1u << chanDesc.w ) - 1 );
                                    break;
                                case CudaTextureAlphaMode::RGB_INTENSITY:
                                    opacityCutoff = ( float )( ( ( ( 1u << chanDesc.x ) - 1 ) + ( ( 1u << chanDesc.y - 1 ) ) + ( ( 1u << chanDesc.z - 1 ) ) ) / 3 );
                                    break;
                                default:
                                    opacityCutoff = 1.f;
                                    break;
                                }
                            }
                        }

                        CudaTextureConfig config;
                        config.opacityCutoff = opacityCutoff;
                        config.transparencyCutoff = transparencyCutoff;
                        config.chanDesc = chanDesc;
                        config.alphaMode = alphaMode;
                        config.texDesc = texDesc;
                        config.width = extent.width;
                        config.height = extent.height;
                        config.depth = extent.depth;

                        std::unique_ptr<CudaTexture> cudaTexture( new CudaTexture( 
                            numUniqueTextures,
                            input.textures[i].cuda.texObject,
                            texDesc.addressMode, filterKernelWidthInTexels, config ) );
                        texture.reset( cudaTexture.release() );

                        width = config.width;
                        height = config.height;
                    }
                    break;
                    case TextureType::STATE:
                    {
                        StateTextureConfig config;
                        config.width = input.textures[i].state.width;
                        config.height = input.textures[i].state.height;
                        config.pitchInBits = input.textures[i].state.pitchInBits;

                        // default tightly packed stride of 2 bits
                        if( config.pitchInBits == 0 )
                            config.pitchInBits = 2;

                        const float filterKernelWidthInTexels = input.textures[i].state.filterKernelWidthInTexels;
                        if( filterKernelWidthInTexels < 0.f || std::isnan( filterKernelWidthInTexels ) )
                            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %f for inputs[%zu].textures[%zu].state.filterKernelWidthInTexels. Must be real positive value.", filterKernelWidthInTexels, inputIdx, i ) );

                        if( config.width > maxExtent )
                            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value of %u for inputs[%zu].textures[%zu].state.width must not exceed %u.", config.width, inputIdx, i, maxExtent ) );

                        if( config.height > maxExtent )
                            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value of %u for inputs[%zu].textures[%zu].state.height must not exceed %u.", config.height, inputIdx, i, maxExtent ) );
                                                
                        std::unique_ptr<StateTexture> stateTexture( new StateTexture( numUniqueTextures, ( const uint8_t* )input.textures[i].state.stateBuffer, config ) );
                        texture.reset( stateTexture.release() );

                        width = config.width;
                        height = config.height;
                    }
                    break;
                    default:
                        throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value of %u for inputs[%zu].textures[%zu].type.", (uint32_t)input.textures[i].type, inputIdx, i ) );
                        break;
                    };

                    size_t tempStorageInBytes = 0;
                    cudaError_t error = texture->build( 0, tempStorageInBytes, 0 );

                    if( error == cudaErrorInvalidChannelDescriptor )
                    {
                        throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Unsupported format for inputs[%zu].textures[%zu].cuda.texObject.", inputIdx, i ) );
                    }
                    else
                    {
                        OMM_CUDA_CHECK( error );
                    }

                    satTempStorageInBytes = std::max( satTempStorageInBytes, tempStorageInBytes );

                    texture->aggregateInto( m_satAggregateBuf );

                    numTexels += width * height;

                    m_textureMap[key] = texture;

                    numUniqueTextures++;
                }
            }
        }

        if( Baker::m_options.maximumSizeInBytes == 0 )
        {
            // By default, the omm array is sized proportional to the number of texels.
            Baker::m_options.maximumSizeInBytes = ( unsigned int )std::min<uint64_t>( ( uint64_t )(~0u), numTexels);
        }

        m_textureBuf.setNumElems( numTextureReferences );

        m_satTempBuf.setNumBytes( satTempStorageInBytes ).setAlignmentInBytes( SAT_TEMP_BUFFER_ALIGNMENT_IN_BYTES );

        m_outOmmIndexBuffers.resize( numInputs );
        m_outOmmUsageDescs.resize( numInputs );

        uint64_t totalNumTriangles = 0;
        for( uint32_t i = 0; i < numInputs; ++i )
            totalNumTriangles += getNumTriangles( inputs[i] );

        // pick the smallest index size that is guarenteed to fit all omms without overlapping the predefined indices
        uint32_t indexSizeInBytes = 0;
        if( totalNumTriangles < ( uint16_t )( -4 ) )
        {
            m_indexFormat = IndexFormat::I16_UINT;
            indexSizeInBytes = sizeof( uint16_t );
        }
        else if( totalNumTriangles < ( uint32_t )( -4 ) )
        {
            m_indexFormat = IndexFormat::I32_UINT;
            indexSizeInBytes = sizeof( uint32_t );
        }
        else
        {
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid number of triangles %lu. Number of triangles must not exceed %u.", totalNumTriangles, ( uint32_t )( -4 ) ) );
        }

        for( uint32_t i = 0; i < numInputs; ++i )
        {
            uint32_t sizeInBytes = getNumTriangles( inputs[i] ) * indexSizeInBytes;
            m_outOmmIndexBuffers[i].setNumBytes( sizeInBytes );
            m_outOmmUsageDescs[i].setNumElems( OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL + 1 );
        }

        m_inputBuf.setNumElems( numInputs );

        m_numTriangles = totalNumTriangles;
        m_inIdBuf.setNumElems( m_numTriangles );
        m_outIdBuf.setNumElems( m_numTriangles );
        m_inHashBuf.setNumElems( m_numTriangles );
        m_outHashBuf.setNumElems( m_numTriangles );

        m_inMarkersBuf.setNumElems( m_numTriangles );
        m_outAssignmentBuf.setNumElems( m_numTriangles );

        // conservative upper bound at minimum of 1 byte per omm
        m_maxNumOmms = std::min( Baker::m_options.maximumSizeInBytes, m_numTriangles );
        m_ommIdBuf.setNumElems( m_maxNumOmms );
        m_sumAreaBuf.setNumElems( 1 );

        m_ommAreaBuf.setNumElems( m_maxNumOmms );
        m_descBuf.setNumElems( m_maxNumOmms );
        m_numOmmsBuf.setNumElems( 1 );
        m_histogramBuf.setNumElems( OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL+1 );
        m_sizeInBytesBuf.setNumElems( 1 );
                
        m_dataBuf.setNumElems( ( Baker::m_options.maximumSizeInBytes + sizeof( uint64_t ) - 1 ) / sizeof( uint64_t ) );

        {
            size_t tempSizeInBytes = 0;
            cudaError_t error = SortPairs<uint32_t, TriangleID>()( 0, tempSizeInBytes, 0, 0, 0, 0, m_numTriangles, 0, sizeof( uint32_t ) * 8, 0 );
            OMM_CUDA_CHECK( error );

            m_sortTempBuf.setNumBytes( tempSizeInBytes ).setAlignmentInBytes( CUB_TEMP_BUFFER_ALIGNMENT_IN_BYTES );
        }

        {
            size_t tempSizeInBytes = 0;
            cudaError_t error = InclusiveSum<uint32_t*, uint32_t*>()( 0, tempSizeInBytes, 0, 0, m_numTriangles );
            OMM_CUDA_CHECK( error );

            m_sumTempBuf.setNumBytes( tempSizeInBytes ).setAlignmentInBytes( CUB_TEMP_BUFFER_ALIGNMENT_IN_BYTES );
        }

        {
            size_t tempSizeInBytes = 0;
            cudaError_t error = ReduceRoundUp<float*, float*, float>()( 0, tempSizeInBytes, 0, 0, m_numTriangles );
            OMM_CUDA_CHECK( error );

            m_reduceTempBuf.setNumBytes( tempSizeInBytes ).setAlignmentInBytes( CUB_TEMP_BUFFER_ALIGNMENT_IN_BYTES );
        }

        {
            size_t tempSizeInBytes = 0;
            cudaError_t error = launchGenerateStartOffsets( 0, tempSizeInBytes, 0, 0, m_maxNumOmms, m_options.format, 0 );
            OMM_CUDA_CHECK( error );

            m_offsetTempBuf.setNumBytes( tempSizeInBytes ).setAlignmentInBytes( CUB_TEMP_BUFFER_ALIGNMENT_IN_BYTES );
        }

        /* Visualization of buffer usage in the different phases of baking.
        *  Buffers with disjoint lifetimes can overlay, reducing the required temporary memory.
        * 
        *                                                        1   2   3   4   5   6   7   8   9  10  11  12
        *                    Element size    Num elements        .   .   .   .   .   .   .   .   .   .   .   .
        *                                                        .   .   .   .   .   .   .   .   .   .   .   .
        * ** Temp Buffers **                                     .   .   .   .   .   .   .   .   .   .   .   .
        * m_inIdBuf          64b             nTri                .   .   xxxxx   .   .   .   .   .   .   .   .
        * m_outIdBuf         64b             nTri                .   .   .   xxxxxxxxxxxxx   .   .   .   .   .
        * m_inHashBuf        32b             nTri                .   .   xxxxx   .   .   .   .   .   .   .   .
        * m_outHashBuf       32b             nTri                .   .   .   xxxxx   .   .   .   .   .   .   .
        * m_inMarkersBuf     32b             nTri                .   .   .   .   xxxxx   .   .   .   .   .   .
        * m_outAssignmentBuf 32b             nTri                .   .   .   .   .   xxxxx   .   .   .   .   .
        * m_ommIdBuf         64b             min( nByte, nTri )  .   .   .   .   .   .   xxxxxxxxxxxxxxxxxxxxx
        * m_sumAreaBuf       32b             1                   .   .   .   .   .   .   .   xxxxx   .   .   .
        * m_ommAreaBuf       32b             min( nByte, nTri )  .   .   .   .   .   .   xxxxxxxxx   .   .   .
        * m_inputBuf                                             .   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        * m_textureBuf                                           .   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        * m_sortTempBuf                                          .   .   .   x   .   .   .   .   .   .   .   .
        * m_sumTempBuf                                           .   .   .   .   .   x   .   .   .   .   .   .
        * m_reduceTempBuf                                        .   .   .   .   .   .   .   x   .   .   .   .
        * m_offsetTempBuf                                        .   .   .   .   .   .   .   .   .   x   .   .
        * m_satTempBuf       32b             nTexels             x   .   .   .   .   .   .   .   .   .   .   .
        * m_satAggregateBuf  64b             nTexels             xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        *                                                        .   .   .   .   .   .   .   .   .   .   .   .
        * ** Output Buffers  **                                  .   .   .   .   .   .   .   .   .   .   .   .
        * m_descBuf          64b             min( nByte, nTri )  .   .   .   .   .   .   .   .   xxxxxxxxxxxxxxx
        * m_dataBuf          8b              nByte               .   .   .   .   .   .   .   .   .   .   .   xxx
        * 
        */

        m_overlay[0]
            .overlay( m_inIdBuf )
            .overlay( m_inMarkersBuf )
            .overlay( m_ommIdBuf );

        m_overlay[1]
            .overlay( m_inHashBuf )
            .overlay( m_ommAreaBuf );

        m_overlay[2]
            .overlay( m_outHashBuf )
            .overlay( m_outAssignmentBuf );

        m_overlay[3]
            .overlay( m_satTempBuf )
            .overlay( m_sortTempBuf )
            .overlay( m_sumTempBuf )
            .overlay( m_reduceTempBuf )
            .overlay( m_offsetTempBuf );

        m_outOmmArrayData.aggregate( m_dataBuf );
        m_outOmmHistogram.aggregate( m_histogramBuf );
        
        // Overlap a temporary buffer with an output buffer to save memory.
        // This results in larger memory requirements for the descriptor buffer when the 
        // omm array size limit constrains the maximum number of omms in the array.
        m_outOmmDesc
            .overlay( m_descBuf )
            .overlay( m_outIdBuf );

        // must match struct PostBakeInfo layout
        m_postBakeInfo.aggregate( m_numOmmsBuf ).aggregate( m_sizeInBytesBuf );

        m_temp
            .aggregate( m_overlay[0] )
            .aggregate( m_overlay[1] )
            .aggregate( m_overlay[2] )
            .aggregate( m_overlay[3] )
            .aggregate( m_textureBuf )
            .aggregate( m_inputBuf )
            .aggregate( m_satAggregateBuf )
            .aggregate( m_sumAreaBuf );

        // if post bake info is disabled, aggregate into the temp buffer.
        if( ( m_options.flags & BakeFlags::ENABLE_POST_BAKE_INFO ) == BakeFlags::ENABLE_POST_BAKE_INFO )
            m_outPostBakeInfo.aggregate( m_postBakeInfo );
        else
            m_temp.aggregate( m_postBakeInfo );

        if( buffers )
        {
            static_assert( std::alignment_of<OptixOpacityMicromapUsageCount>::value <= BufferAlignmentInBytes::MICROMAP_USAGE_COUNTS, "alignment must at least match the element type" );
            static_assert( std::alignment_of<OptixOpacityMicromapDesc>::value <= BufferAlignmentInBytes::MICROMAP_DESC, "alignment must at least match the element type" );
            static_assert( std::alignment_of<OptixOpacityMicromapHistogramEntry>::value <= BufferAlignmentInBytes::MICROMAP_HISTOGRAM_ENTRIES, "alignment must at least match the element type" );
            static_assert( std::alignment_of<PostBakeInfo>::value <= BufferAlignmentInBytes::POST_BAKE_INFO, "alignment must at least match the element type" );

            if( buffers->indexFormat != m_indexFormat )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for buffers->indexFormat. Expected format %u.", (uint32_t) buffers->indexFormat, (uint32_t) m_indexFormat ) );

            if( buffers->numMicromapDescs && buffers->perMicromapDescBuffer == 0 )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for buffers->perMicromapDescBuffer. Must not be zero." ) );
            if( buffers->numMicromapHistogramEntries && buffers->micromapHistogramEntriesBuffer == 0 )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for buffers->micromapHistogramEntriesBuffer. Must not be zero." ) );
            if( buffers->outputBufferSizeInBytes && buffers->outputBuffer == 0 )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for buffers->outputBuffer. Must not be zero." ) );

            if( buffers->perMicromapDescBuffer & ( BufferAlignmentInBytes::MICROMAP_DESC - 1 ) )
                throw Exception( Result::ERROR_MISALIGNED_ADDRESS, stringf( "Invalid value %p for buffers->perMicromapDescBuffer. Must be %zu byte aligned.", ( void* )buffers->perMicromapDescBuffer, ( size_t )BufferAlignmentInBytes::MICROMAP_DESC ) );
            if( buffers->micromapHistogramEntriesBuffer & ( BufferAlignmentInBytes::MICROMAP_HISTOGRAM_ENTRIES - 1 ) )
                throw Exception( Result::ERROR_MISALIGNED_ADDRESS, stringf( "Invalid value %p for buffers->micromapHistogramEntriesBuffer. Must be %zu byte aligned.", ( void* )buffers->micromapHistogramEntriesBuffer, ( size_t )BufferAlignmentInBytes::MICROMAP_HISTOGRAM_ENTRIES ) );
            if( buffers->outputBuffer & ( BufferAlignmentInBytes::MICROMAP_OUTPUT - 1 ) )
                throw Exception( Result::ERROR_MISALIGNED_ADDRESS, stringf( "Invalid value %p for buffers->outputBuffer. Must be %zu byte aligned.", ( void* )buffers->outputBuffer, ( size_t )BufferAlignmentInBytes::MICROMAP_OUTPUT ) );

            if( buffers->tempBufferSizeInBytes && buffers->tempBuffer == 0 )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for buffers->tempBuffer. Must not be zero." ) );

            if( ( m_options.flags & BakeFlags::ENABLE_POST_BAKE_INFO ) == BakeFlags::ENABLE_POST_BAKE_INFO )
            {
                if( buffers->postBakeInfoBuffer == 0 )
                    throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for buffers->postBakeInfoBuffer. Must not be zero." ) );
                if( buffers->postBakeInfoBuffer & ( BufferAlignmentInBytes::POST_BAKE_INFO - 1 ) )
                    throw Exception( Result::ERROR_MISALIGNED_ADDRESS, stringf( "Invalid value %p for buffers->postBakeInfoBuffer. Must be %zu byte aligned.", ( void* )buffers->postBakeInfoBuffer, ( size_t )BufferAlignmentInBytes::POST_BAKE_INFO ) );
            }
            else
            {
                if( buffers->postBakeInfoBufferSizeInBytes )
                    throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %zu for buffers->postBakeInfoBufferSizeInBytes. Must be zero when BakeFlags::ENABLE_POST_BAKE_INFO flag is not set.", buffers->postBakeInfoBufferSizeInBytes ) );
                if( buffers->postBakeInfoBuffer )
                    throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %llu for buffers->postBakeInfoBuffer. Must be zero when BakeFlags::ENABLE_POST_BAKE_INFO flag is not set.", buffers->postBakeInfoBuffer ) );
            }

            m_outOmmArrayData.materialize( ( unsigned char* )buffers->outputBuffer );
            m_outOmmDesc.materialize( ( OptixOpacityMicromapDesc* )buffers->perMicromapDescBuffer );
            m_outOmmHistogram.materialize( ( OptixOpacityMicromapHistogramEntry* )buffers->micromapHistogramEntriesBuffer );
            m_outPostBakeInfo.materialize( ( unsigned char* )buffers->postBakeInfoBuffer );
            m_temp.materialize( ( unsigned char* )buffers->tempBuffer );

            if( ( m_options.flags & BakeFlags::ENABLE_POST_BAKE_INFO ) == BakeFlags::ENABLE_POST_BAKE_INFO )
            {
                if( m_outPostBakeInfo.getNumBytes() > buffers->postBakeInfoBufferSizeInBytes )
                    throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %zu for buffers->postBakeInfoBufferSizeInBytes. Must be at least %zu bytes.", buffers->postBakeInfoBufferSizeInBytes, m_outPostBakeInfo.getNumBytes() ) );
            }

            if( m_outOmmArrayData.getNumBytes() > buffers->outputBufferSizeInBytes )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %zu for buffers->outputBufferSizeInBytes. Must be at least %zu bytes.", buffers->outputBufferSizeInBytes, m_outOmmArrayData.getNumBytes() ) );
            if( m_outOmmDesc.getNumElems() > buffers->numMicromapDescs )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %zu for buffers->numMicromapDescs. Must be at least %zu.", buffers->numMicromapDescs, m_outOmmDesc.getNumElems() ) );
            if( m_outOmmHistogram.getNumElems() > buffers->numMicromapHistogramEntries )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %zu for buffers->numMicromapHistogramEntries. Must be at least %zu.", buffers->numMicromapHistogramEntries, m_outOmmHistogram.getNumElems() ) );
            if( m_temp.getNumBytes() > buffers->tempBufferSizeInBytes )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %zu for buffers->tempBufferSizeInBytes. Must be at least %zu bytes.", buffers->tempBufferSizeInBytes, m_temp.getNumBytes() ) );
        }
        else
        {
            m_outOmmArrayData.materialize();
            m_outOmmDesc.materialize();
            m_outOmmHistogram.materialize();
            m_outPostBakeInfo.materialize();
            m_temp.materialize();
        }

        if( inputBuffers )
        {
            for( size_t i = 0; i < m_outOmmIndexBuffers.size(); ++i )
            {
                if( inputBuffers[i].indexBufferSizeInBytes && inputBuffers[i].indexBuffer == 0 )
                    throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputBuffers[%zu]->outOmmIndex. Must not be zero.", i ) );

                if( inputBuffers[i].indexBuffer & ( indexSizeInBytes - 1) )
                    throw Exception( Result::ERROR_MISALIGNED_ADDRESS, stringf( "Invalid value %llu for inputBuffers[%zu]->outOmmIndex. Must be %u byte aligned.", inputBuffers[i].indexBuffer, i, indexSizeInBytes ) );

                m_outOmmIndexBuffers[i].materialize( ( unsigned char* )inputBuffers[i].indexBuffer );

                if( m_outOmmIndexBuffers[i].getNumBytes() > inputBuffers[i].indexBufferSizeInBytes )
                    throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %zu for inputBuffers[%zu].indexBufferSizeInBytes. Must be at least %zu.", inputBuffers[i].indexBufferSizeInBytes, i, m_outOmmIndexBuffers[i].getNumBytes() ) );
            }

            for( size_t i = 0; i < m_outOmmUsageDescs.size(); ++i )
            {
                if( inputBuffers[i].numMicromapUsageCounts && inputBuffers[i].micromapUsageCountsBuffer == 0 )
                    throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputBuffers[%zu]->outOmmIndex. Must not be zero.", i ) );

                if( inputBuffers[i].micromapUsageCountsBuffer & ( BufferAlignmentInBytes::MICROMAP_USAGE_COUNTS - 1 ) )
                    throw Exception( Result::ERROR_MISALIGNED_ADDRESS, stringf( "Invalid value %p for inputBuffers[%zu].micromapUsageCountsBuffer. Must be %zu byte aligned.", ( void* )inputBuffers[i].micromapUsageCountsBuffer, i, ( size_t )BufferAlignmentInBytes::MICROMAP_USAGE_COUNTS ) );

                m_outOmmUsageDescs[i].materialize( ( OptixOpacityMicromapUsageCount* )inputBuffers[i].micromapUsageCountsBuffer );

                if( m_outOmmUsageDescs[i].getNumElems() > inputBuffers[i].numMicromapUsageCounts )
                    throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %zu for inputBuffers[%zu].numMicromapUsageCounts. Must be at least %zu.", inputBuffers[i].numMicromapUsageCounts, i, m_outOmmUsageDescs[i].getNumElems() ) );
            }
        }
        else
        {
            for( size_t i = 0; i < m_outOmmIndexBuffers.size(); ++i )
                m_outOmmIndexBuffers[i].materialize();

            for( size_t i = 0; i < m_outOmmUsageDescs.size(); ++i )
                m_outOmmUsageDescs[i].materialize();
        }
    }

    BakeBuffers getPreBakeInfo()
    {
        BakeBuffers preBake = {};
        preBake.indexFormat = m_indexFormat;
        preBake.outputBufferSizeInBytes = m_outOmmArrayData.getNumBytes();
        preBake.numMicromapDescs = m_outOmmDesc.getNumElems();
        preBake.numMicromapHistogramEntries = m_outOmmHistogram.getNumElems();
        preBake.postBakeInfoBufferSizeInBytes = m_outPostBakeInfo.getNumBytes();
        preBake.tempBufferSizeInBytes = m_temp.getNumBytes() + (m_temp.getAlignmentInBytes() - 1); // add padding so realigning user temp input won't cause an overflow

        return preBake;
    }

    BakeInputBuffers getPreBakeInputInfo( unsigned int index )
    {
        BakeInputBuffers preBake = {};
        preBake.indexBufferSizeInBytes = m_outOmmIndexBuffers[index].getNumBytes();
        preBake.numMicromapUsageCounts = m_outOmmUsageDescs[index].getNumElems();
        return preBake;
    }

    void execute( cudaStream_t stream )
    {
        uint32_t maxOmmArraySizeInBytes = m_dataBuf.getNumBytes();

        int device;
        OMM_CUDA_CHECK( cudaGetDevice( &device ) );

        cudaDeviceProp props;
        OMM_CUDA_CHECK( cudaGetDeviceProperties( &props, device ) );

        uint32_t numThreads = props.multiProcessorCount * 512u;

        // 1. Build texture summed area tables.

        // build opacity summed area tables per texture
        for( auto itr : m_textureMap )
        {
            size_t tempStorageInBytes = m_satTempBuf.getNumBytes();

            cudaError_t error = itr.second->build( m_satTempBuf.access(), tempStorageInBytes, stream );
            OMM_CUDA_CHECK( error );

            // dump( "sat:", itr.second.satBuf );
        }

        // 2. Upload bake and texture input descriptors.

        std::vector<TextureInput> textureInputs;
        std::vector<BakeInput> bakeInputs;
        bakeInputs.resize( m_inputs.size() );
        textureInputs.resize( m_textureBuf.getNumElems() );

        uint32_t textureInputOffset = 0;
        for( uint32_t i = 0; i < m_inputs.size(); ++i )
        {
            bakeInputs[i].desc           = m_inputs[i];
            bakeInputs[i].outAssignments = m_outOmmIndexBuffers[i].access();
            bakeInputs[i].inTextures     = m_textureBuf.access() + textureInputOffset;

            for( uint32_t j = 0; j < m_inputs[i].numTextures; ++j )
            {
                const auto& texture      = m_inputs[i].textures[j];
                const auto& textureDesc  = m_textureMap[texture];
                auto& textureInput = textureInputs[textureInputOffset + j];
                textureInput.data = textureDesc->get( texture );

                // triangles are matched with a 1% texel width accuracy
                const float quantizationEpsilonInTexels = 0.01f;

                float2 quantizationFrequencyInUV = make_float2(
                    textureInput.data.width / quantizationEpsilonInTexels,
                    textureInput.data.height / quantizationEpsilonInTexels );

                float2 periodInUV = { 1.f, 1.f };

                auto addressModeToPeriod = []( cudaTextureAddressMode addressMode ) {
                    switch( addressMode )
                    {
                        break;
                    case cudaAddressModeWrap:
                        return 1.f;
                        break;
                    case cudaAddressModeMirror:
                        return 2.f;
                        break;
                    default:
                        return 0.f;
                    }
                };

                periodInUV.x *= addressModeToPeriod( textureInput.data.addressMode[0] );
                periodInUV.y *= addressModeToPeriod( textureInput.data.addressMode[1] );

                textureInput.quantizationFrequency = quantizationFrequencyInUV;

                textureInput.quantizationPeriod.x = textureInput.quantizationFrequency.x ? ( 1.f / textureInput.quantizationFrequency.x ) : 0;
                textureInput.quantizationPeriod.y = textureInput.quantizationFrequency.y ? ( 1.f / textureInput.quantizationFrequency.y ) : 0;

                textureInput.quantizedPeriod.x = periodInUV.x * textureInput.quantizationFrequency.x;
                textureInput.quantizedPeriod.y = periodInUV.y * textureInput.quantizationFrequency.y;

                textureInput.quantizedFrequency.x = textureInput.quantizedPeriod.x ? ( 1.f / textureInput.quantizedPeriod.x ) : 0.f;
                textureInput.quantizedFrequency.y = textureInput.quantizedPeriod.y ? ( 1.f / textureInput.quantizedPeriod.y ) : 0.f;
            }

            textureInputOffset += m_inputs[i].numTextures;
        }

        CudaMemcpyAsync( m_textureBuf, textureInputs, stream );
        CudaMemcpyAsync( m_inputBuf, bakeInputs, stream );

        // 3. Setup triangles, detect uniforms and generate hashes for duplicate detection.

        uint32_t triangleOffset = 0;
        for( uint32_t i = 0; i < m_inputs.size(); ++i )
        {
            SetupBakeInputParams params = {};

            params.numTriangles   = getNumTriangles( m_inputs[i] );
            params.inputIdx       = i;
            params.outTriangleIDs = m_inIdBuf.access() + triangleOffset;
            params.outHashKeys    = m_inHashBuf.access() + triangleOffset;
            params.textures       = bakeInputs[i].inTextures;
            params.input          = m_inputs[i];
            params.format         = m_options.format;

            OMM_CUDA_CHECK( launchSetupBakeInput( params, stream ) );

            triangleOffset += params.numTriangles;
        }

        // 4. Sort triangles by their hash keys

        {
            // sort triangles by hash key
            size_t tempSizeInBytes = m_sortTempBuf.getNumBytes();
            cudaError_t error = SortPairs<uint32_t, TriangleID>()(
                m_sortTempBuf.access(), tempSizeInBytes, m_inHashBuf.access(), m_outHashBuf.access(), m_inIdBuf.access(), m_outIdBuf.access(), m_numTriangles, 0, sizeof( uint32_t ) * 8, stream );
            OMM_CUDA_CHECK( error );
        }

        // 5. Find and mark the start of duplicate groups in the sorted triangle list.

        {
            MarkFirstOmmOccuranceParams params;
            params.numTriangles = m_numTriangles;
            params.inHashKeys = m_outHashBuf.access();
            params.inTriangleIDs = m_outIdBuf.access();
            params.outMarkers = m_inMarkersBuf.access();
            params.inBakeInputs = m_inputBuf.access();

            OMM_CUDA_CHECK( launchMarkFirstOmmOccurance( params, stream ) );
        }

        // 6. Generate flat omm assignment

        {
            size_t tempSizeInBytes = m_sumTempBuf.getNumBytes();
            cudaError_t error = InclusiveSum<uint32_t*, uint32_t*>()(
                m_sumTempBuf.access(), tempSizeInBytes, m_inMarkersBuf.access(), m_outAssignmentBuf.access(), m_numTriangles, stream );
            OMM_CUDA_CHECK( error );
        }

        // 7. Scatter the assignments into the per-input assigment buffers. Output omm area.

        CudaMemsetAsync( m_ommAreaBuf, stream );

        {
            GenerateAssignmentParams params;
            params.numTriangles = m_numTriangles;
            params.inTriangleIDs = m_outIdBuf.access();
            params.maxOmms = m_maxNumOmms;
            params.outNumOmms = m_numOmmsBuf.access();
            params.inAssignment = m_outAssignmentBuf.access();
            params.outOmmTriangleId = m_ommIdBuf.access();
            params.outOmmArea = m_ommAreaBuf.access();
            params.inBakeInputs = m_inputBuf.access();
            params.indexFormat = m_indexFormat;

            OMM_CUDA_CHECK( launchGenerateAssignment( params, stream ) );
        }

        // 8. Sum the omm area. Use roundup mode to compute a conservative upper bound on the sum.
        // Rounding up is necceseary to prevent overflows in the assignment.

        {
            size_t tempSizeInBytes = m_reduceTempBuf.getNumBytes();
            cudaError_t error = ReduceRoundUp<float*, float*, float>()( m_reduceTempBuf.access(), tempSizeInBytes, m_ommAreaBuf.access(), m_sumAreaBuf.access(), m_numTriangles, stream );
            OMM_CUDA_CHECK( error );
        }

        // 9. Generate omm descriptors, assign subdivision levels, compute total omm array size and subdivision level histogram.
        
        CudaMemsetAsync( m_sizeInBytesBuf, stream );

        // initialize the histogram
        std::vector<OptixOpacityMicromapHistogramEntry> histogram( m_histogramBuf.getNumElems(), OptixOpacityMicromapHistogramEntry { 0, 0, m_options.format } );
        for( size_t i = 0; i < histogram.size(); ++i )
            histogram[i].subdivisionLevel = i;
        CudaMemcpyAsync( m_histogramBuf, histogram, stream );

        // assign subdivision levels to omms
        {
            GenerateLayoutParams params;
            params.inOmmArea = m_ommAreaBuf.access();
            params.inSumArea = m_sumAreaBuf.access();
            params.inNumOmms = m_numOmmsBuf.access();
            params.ioDescs = m_descBuf.access();
            params.maxOmmArraySizeInBytes = maxOmmArraySizeInBytes;
            params.ioSizeInBytes = m_sizeInBytesBuf.access();
            params.ioHistogram = m_histogramBuf.access();
            params.microTrianglesPerTexel = ( m_options.subdivisionScale != 0.f ) ? ( 1.f / ( m_options.subdivisionScale * m_options.subdivisionScale ) ) : 0.f;
            params.format = m_options.format;

            OMM_CUDA_CHECK( launchGenerateLayout( params, m_numTriangles, stream ) );
        }

        // 10. Generate omm desc byte offsets by summing over the omm sizes in bytes.

        {
            size_t temp_storage_bytes = m_offsetTempBuf.getNumBytes();
            OMM_CUDA_CHECK( launchGenerateStartOffsets( m_offsetTempBuf.access(), temp_storage_bytes, m_descBuf.access(), m_descBuf.access(), m_descBuf.getNumElems(), m_options.format, stream ) );
        }

        // 11. Generate per-input usage histograms.

        std::vector<OptixOpacityMicromapUsageCount> usage( m_outOmmUsageDescs[0].getNumElems(), OptixOpacityMicromapUsageCount{ 0, 0, m_options.format } );
        for( size_t i = 0; i < usage.size(); ++i )
            usage[i].subdivisionLevel = i;

        for( uint32_t i = 0; i < m_inputs.size(); ++i )
        {
            CudaMemcpyAsync( m_outOmmUsageDescs[i], usage, stream );

            GenerateInputHistogramParams params;
            params.indexFormat  = m_indexFormat;
            params.numTriangles = getNumTriangles( m_inputs[i] );
            params.inAssignment = bakeInputs[i].outAssignments;
            params.inDescs      = m_descBuf.access();
            params.ioHistogram  = m_outOmmUsageDescs[i].access();
            OMM_CUDA_CHECK( launchGenerateInputHistogram( params, stream ) );
        }

        // 12. Evaluate the opacity states of all micro triangles in the opacity micromap array

        CudaMemsetAsync( m_dataBuf, stream );

        // evaluate the opacity of all microtriangles in the array
        {
            EvaluateOmmOpacityParams params;
            params.inNumOmms = m_numOmmsBuf.access();
            params.inSizeInBytes = m_sizeInBytesBuf.access();
            params.inDescs = m_descBuf.access();
            params.ioData = m_dataBuf.access();
            params.inTriangleIdPerOmm = m_ommIdBuf.access();
            params.inBakeInputs = m_inputBuf.access();
            params.dataSizeInBytes = m_dataBuf.getNumBytes();
            params.format = m_options.format;

            const uint32_t maxThreads = m_dataBuf.getNumBytes() * ( ( m_options.format == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE ) ? 8 : 4 );

            // use the maximum data size as a limit for the number of threads launched
            OMM_CUDA_CHECK( launchEvaluateOmmOpacity( params, std::min<uint32_t>( numThreads, maxThreads ), stream ) );
        }
    }

private:

    void validate( const BakeOptions& options )
    {
        if( ( options.flags & ~( BakeFlags::ENABLE_POST_BAKE_INFO ) ) != BakeFlags::NONE )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for options.flags. Contains invalid flags.", (uint32_t) options.flags ) );

        if( options.subdivisionScale < 0.f || std::isnan( options.subdivisionScale ) )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %f for options.subdivisionScale. Must be real positive value.", options.subdivisionScale ) );

        switch( options.format )
        {
        case OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE:
        case OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE:
            break;
        default:
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %i for options.format.", options.format ) );
            break;
        }
    }

    void validate( const TextureDesc& texture, uint32_t inputIdx, uint32_t textureIdx, bool isPreBake )
    {
        switch( texture.type )
        {
        case TextureType::CUDA:
            if( texture.cuda.texObject == 0 )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].textures[%u].cuda.texObject. Must not be zero.", inputIdx, textureIdx ) );

            if( std::isnan(texture.cuda.transparencyCutoff) )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %f for inputs[%u].textures[%u].cuda.transparencyCutoff.",
                    texture.cuda.transparencyCutoff, inputIdx, textureIdx ) );

            if( std::isnan( texture.cuda.opacityCutoff ) )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %f for inputs[%u].textures[%u].cuda.opacityCutoff.",
                    texture.cuda.opacityCutoff, inputIdx, textureIdx ) );

            if( texture.cuda.transparencyCutoff > texture.cuda.opacityCutoff )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %f for inputs[%u].textures[%u].cuda.transparencyCutoff. Must not be larger than value %f of inputs[%u].textures[%u].cuda.opacityCutoff.",
                    texture.cuda.transparencyCutoff, inputIdx, textureIdx, texture.cuda.opacityCutoff, inputIdx, textureIdx ) );
            break;
        case TextureType::STATE:
        {
            auto validateAddressMode = [&]( uint32_t dim ) {
                switch( texture.state.addressMode[dim] )
                {
                case cudaAddressModeWrap:
                case cudaAddressModeClamp:
                case cudaAddressModeMirror:
                    break;
                default:
                    throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].textures[%u].state.addressMode[%i].",
                        texture.state.addressMode[dim], inputIdx, textureIdx, dim ) );
                }
            };

            validateAddressMode( 0 );
            validateAddressMode( 1 );

            if( texture.state.width == 0 )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value of for inputs[%u].textures[%u].state.width. Must not be zero.", inputIdx, textureIdx ) );

            if( texture.state.height == 0 )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value of for inputs[%u].textures[%u].state.height. Must not be zero", inputIdx, textureIdx ) );
            
            if( !isPreBake && texture.state.stateBuffer == 0 )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].textures[%u].state.stateBuffer. Must not be zero.", inputIdx, textureIdx ) );

            if( ( texture.state.pitchInBits & 1 ) != 0 )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].textures[%u].state.pitchInBits. Must be 2 bit aligned.", texture.state.pitchInBits, inputIdx, textureIdx ) );

            break;
        }
        default:
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].textures[%u].type", ( uint32_t )texture.type, inputIdx, textureIdx ) );
        };
    }

    void validate( const BakeInputDesc& input, uint32_t inputIdx, bool isPreBake )
    {
        uint64_t coordAlignmentInBytes = 0;
        switch( input.texCoordFormat )
        {
        case TexCoordFormat::UV32_FLOAT2:
            coordAlignmentInBytes = 4;
            break;
        default:
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].texCoordFormat.", ( uint32_t )input.texCoordFormat, inputIdx ) );
        };

        if( !isPreBake && input.texCoordBuffer == 0 )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].texCoordBuffer. Must not be zero.", inputIdx ) );

        if( ( input.texCoordBuffer & ( coordAlignmentInBytes - 1 ) ) != 0 )
            throw Exception( Result::ERROR_MISALIGNED_ADDRESS, stringf( "Invalid value %p for inputs[%u].texCoordBuffer. Must be %u byte aligned.", ( void* )input.texCoordBuffer, inputIdx, ( uint32_t )coordAlignmentInBytes ) );

        if( ( input.texCoordStrideInBytes & ( coordAlignmentInBytes - 1 ) ) != 0 )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].texCoordStrideInBytes. Must be %u byte aligned.", input.texCoordStrideInBytes, inputIdx, ( uint32_t )coordAlignmentInBytes ) );

        uint64_t indexAlignmentInBytes = 0;
        switch( input.indexFormat )
        {
        case IndexFormat::NONE:
            indexAlignmentInBytes = 0;
            break;
        case IndexFormat::I8_UINT:
            indexAlignmentInBytes = 1;
            break;
        case IndexFormat::I16_UINT:
            indexAlignmentInBytes = 2;
            break;
        case IndexFormat::I32_UINT:
            indexAlignmentInBytes = 4;
            break;
        default:
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].indexFormat.", (uint32_t) input.indexFormat, inputIdx ) );
        };

        if( !isPreBake && input.indexBuffer == 0 && input.indexFormat != IndexFormat::NONE )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].indexBuffer. Must not be zero when inputs[%u].indexFormat is not NONE.", inputIdx, inputIdx ) );
        else if( input.indexBuffer != 0 && input.indexFormat == IndexFormat::NONE )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].indexBuffer. Must be zero when inputs[%u].indexFormat is NONE.", inputIdx, inputIdx ) );

        if( ( input.indexBuffer & ( indexAlignmentInBytes - 1 ) ) != 0 )
            throw Exception( Result::ERROR_MISALIGNED_ADDRESS, stringf( "Invalid value %p for inputs[%u].indexBuffer. Must be %u byte aligned.", ( void* )input.indexBuffer, inputIdx, ( uint32_t )indexAlignmentInBytes ) );

        if( ( input.indexTripletStrideInBytes & ( indexAlignmentInBytes - 1 ) ) != 0 )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].indexTripletStrideInBytes. Must be %u byte aligned.", input.indexTripletStrideInBytes, inputIdx, ( uint32_t )indexAlignmentInBytes ) );

        const uint32_t maxTriangleCount = ( 1 << 28 );
        if( input.indexFormat != IndexFormat::NONE )
        {
            if( input.numIndexTriplets > maxTriangleCount )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].numIndexTriplets. Must not exceed %u.", input.numIndexTriplets, inputIdx, maxTriangleCount ) );

            if( input.numIndexTriplets == 0 )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].numIndexTriplets. The value must be non-zero when inputs[%u].indexFormat is not NONE.", inputIdx, inputIdx ) );

            if( input.numTexCoords >= ( 1u << 8 ) && input.indexFormat == IndexFormat::I8_UINT )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].numTexCoords. inputs[%u].numTexCoords must not exceed 255 when inputs[%u].indexFormat is I8_UINT.", ( uint32_t )input.numTexCoords, inputIdx, inputIdx, inputIdx ) );
            if( input.numTexCoords >= ( 1u << 16 ) && input.indexFormat == IndexFormat::I16_UINT )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].numTexCoords. inputs[%u].numTexCoords must not exceed 65535 when inputs[%u].indexFormat is I16_UINT.", ( uint32_t )input.numTexCoords, inputIdx, inputIdx, inputIdx ) );
        }
        else
        {
            if( input.numIndexTriplets != 0 )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].numIndexTriplets. The value must be zero when inputs[%u].indexFormat is NONE.", inputIdx, inputIdx ) );

            if( input.numTexCoords == 0 )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].numTexCoords value. The value must be non-zero when inputs[%u].indexFormat is NONE.", inputIdx, inputIdx ) );

            if( ( input.numTexCoords % 3 ) != 0 )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].numTexCoords. The value must be multiple of 3 when inputs[%u].indexFormat is NONE.", input.numTexCoords, inputIdx, inputIdx ) );

            if( input.numTexCoords > 3 * maxTriangleCount )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].numTexCoords. Must not exceed %u.", input.numTexCoords, inputIdx, 3 * maxTriangleCount ) );
        }

        uint64_t textureAlignmentInBytes = 0;
        switch( input.textureIndexFormat )
        {
        case IndexFormat::NONE:
            textureAlignmentInBytes = 0;
            break;
        case IndexFormat::I8_UINT:
            textureAlignmentInBytes = 1;
            break;
        case IndexFormat::I16_UINT:
            textureAlignmentInBytes = 2;
            break;
        case IndexFormat::I32_UINT:
            textureAlignmentInBytes = 4;
            break;
        default:
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].textureIndexFormat.", (uint32_t)input.textureIndexFormat, inputIdx ) );
        };

        if( !isPreBake && input.textureIndexBuffer == 0 && input.textureIndexFormat != IndexFormat::NONE )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].textureIndexBuffer. Must not be zero when inputs[%u].textureIndexFormat is not NONE.", inputIdx, inputIdx ) );
        else if( input.textureIndexBuffer != 0 && input.textureIndexFormat == IndexFormat::NONE )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].textureIndexBuffer. Must be zero when inputs[%u].textureIndexFormat is NONE.", inputIdx, inputIdx ) );

        if( ( input.textureIndexBuffer & ( textureAlignmentInBytes - 1 ) ) != 0 )
            throw Exception( Result::ERROR_MISALIGNED_ADDRESS, stringf( "Invalid value %p for inputs[%u].textureIndexBuffer. Must be %u byte aligned.", ( void* )input.textureIndexBuffer, inputIdx, ( uint32_t )textureAlignmentInBytes ) );

        if( ( input.textureIndexStrideInBytes & ( textureAlignmentInBytes - 1 ) ) != 0 )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].textureIndexStrideInBytes. Must be %u byte aligned.", input.textureIndexStrideInBytes, inputIdx, ( uint32_t )textureAlignmentInBytes ) );

        if( input.textures == 0 )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].textures. Must not be zero.", inputIdx ) );

        if( input.numTextures == 0 )
        {
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].numTextures. Must not be zero.", inputIdx ) );
        }
        else if( input.numTextures > 1 )
        {
            if( input.textureIndexFormat == IndexFormat::NONE )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].textureIndexFormat. Must not be NONE when inputs[%u].numTextures is larger than one.", inputIdx, inputIdx ) );

            if( input.numTextures >= (1u << 8) && input.textureIndexFormat == IndexFormat::I8_UINT )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].numTextures. inputs[%u].numTextures must not exceed 255 when inputs[%u].textureIndexFormat is I8_UINT.", (uint32_t)input.numTextures, inputIdx, inputIdx, inputIdx ) );
            if( input.numTextures >= ( 1u << 16 ) && input.textureIndexFormat == IndexFormat::I16_UINT )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].numTextures. inputs[%u].numTextures must not exceed 65535 when inputs[%u].textureIndexFormat is I16_UINT.", ( uint32_t )input.numTextures, inputIdx, inputIdx, inputIdx ) );
        }
        else if( input.numTextures == 1 )
        {
            if( input.textureIndexFormat != IndexFormat::NONE )
                throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].textureIndexFormat. Must be NONE when inputs[%u].numTextures is one.", (uint32_t) input.textureIndexFormat, inputIdx, inputIdx ) );
        }

        for( unsigned int i = 0; i < input.numTextures; ++i )
            validate( input.textures[i], inputIdx, i, isPreBake );

        uint64_t transformAlignmentInBytes = 0;
        switch( input.transformFormat )
        {
        case UVTransformFormat::NONE:
            break;
        case UVTransformFormat::MATRIX_FLOAT2X3:
            transformAlignmentInBytes = 8;
            break;
        default:
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value %u for inputs[%u].transformFormat.", ( uint32_t )input.transformFormat, inputIdx ) );
        }

        if( !isPreBake && input.transform == 0 && input.transformFormat != UVTransformFormat::NONE )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].transform. Must not be zero when inputs[%u].transformFormat is not NONE.", inputIdx, inputIdx ) );
        else if( input.transform != 0 && input.transformFormat == UVTransformFormat::NONE )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputs[%u].transform. Must be zero when inputs[%u].transformFormat is NONE.", inputIdx, inputIdx ) );

        if( ( input.transform & ( transformAlignmentInBytes - 1 ) ) != 0 )
            throw Exception( Result::ERROR_MISALIGNED_ADDRESS, stringf( "Invalid value %p for inputs[%u].transform. Must be %u byte aligned.", ( void* )input.transform, inputIdx, ( uint32_t )transformAlignmentInBytes ) );
    }

    BakeOptions                m_options;
    std::vector<BakeInputDesc> m_inputs;

    uint32_t m_numTriangles;
    
    // conservative upper bound
    uint32_t m_maxNumOmms;

    // format for output omm indices
    IndexFormat m_indexFormat;

    // external per bake-input buffes
    std::vector<BufferLayout<>>                               m_outOmmIndexBuffers;
    std::vector<BufferLayout<OptixOpacityMicromapUsageCount>> m_outOmmUsageDescs;

    // external buffers
    BufferLayout<>                                   m_outOmmArrayData;
    BufferLayout<OptixOpacityMicromapDesc>           m_outOmmDesc;
    BufferLayout<OptixOpacityMicromapHistogramEntry> m_outOmmHistogram;
    BufferLayout<>                                   m_outPostBakeInfo;
    BufferLayout<>                                   m_temp;

    // subbuffers
    BufferLayout<uint64_t>                            m_dataBuf;
    BufferLayout<OptixOpacityMicromapDesc>            m_descBuf;
    BufferLayout<OptixOpacityMicromapHistogramEntry>  m_histogramBuf;
    BufferLayout<uint32_t>                            m_sizeInBytesBuf;
    BufferLayout<uint32_t>                            m_numOmmsBuf;

    // temporary buffers
    BufferLayout<>             m_sortTempBuf;
    BufferLayout<>             m_sumTempBuf;
    BufferLayout<>             m_reduceTempBuf;
    BufferLayout<>             m_offsetTempBuf;
    BufferLayout<>             m_satTempBuf;
    BufferLayout<uint32_t>     m_satAggregateBuf;
    BufferLayout<TriangleID>   m_inIdBuf, m_outIdBuf;
    BufferLayout<uint32_t>     m_inHashBuf, m_outHashBuf;
    BufferLayout<uint32_t>     m_inMarkersBuf, m_outAssignmentBuf;
    BufferLayout<TriangleID>   m_ommIdBuf;
    BufferLayout<float>        m_sumAreaBuf;
    BufferLayout<float>        m_ommAreaBuf;
    BufferLayout<BakeInput>   m_inputBuf;
    BufferLayout<TextureInput> m_textureBuf;

    // aggregate buffer matching the PostBakeInfo struct
    BufferLayout<> m_postBakeInfo;

    // auxiliary buffers to facilitate overlaying of temporary buffers with disjoint lifetimes
    BufferLayout<> m_overlay[4];

    // texture descriptors mapping to summed area table encoding.
    TextureMap m_textureMap;
};


Result GetPreBakeInfo( 
    const BakeOptions*     options,
    unsigned               numInputs,
    const BakeInputDesc*   inputs,
    BakeInputBuffers*      outInputBuffers,
    BakeBuffers*           outBuffers )
{
    try
    {
        if( outInputBuffers == 0 )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for outInputBuffers. Must not be zero." ) );

        if( outBuffers == 0 )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for outBuffers. Must not be zero." ) );

        Baker baker( options, numInputs, inputs, 0, 0 );

        *outBuffers = baker.getPreBakeInfo();

        for( unsigned i = 0; i < numInputs; ++i )
            outInputBuffers[i] = baker.getPreBakeInputInfo( i );
    }
    catch( const Exception& exception )
    {
        std::cerr << exception.what() << std::endl;
        return exception.getResult();
    }
    catch( ... )
    {
        return Result::ERROR_INTERNAL;
    }

    return Result::SUCCESS;
}

Result BakeOpacityMicromaps( 
    const BakeOptions*      options,
    unsigned                numInputs,
    const BakeInputDesc*    inputs,
    const BakeInputBuffers* inputBuffers,
    const BakeBuffers*      buffers,
    cudaStream_t            stream )
{
    try
    {
        if( inputBuffers == 0 )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for inputBuffers. Must not be zero." ) );

        if( buffers == 0 )
            throw Exception( Result::ERROR_INVALID_VALUE, stringf( "Invalid value for buffers. Must not be zero." ) );

        Baker baker( options, numInputs, inputs, inputBuffers, buffers );
        baker.execute( stream );
    }
    catch( const Exception& exception )
    {
        std::cerr << exception.what() << std::endl;
        return exception.getResult();
    }
    catch( ... ) 
    {
        return Result::ERROR_INTERNAL;
    }
    
    return Result::SUCCESS;
}

} // namespace cuOmmBaking
