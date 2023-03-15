#include "Image.h"
#include <cfloat>
#include <climits>
#include <fstream>
#include <iostream>

#ifdef _MSC_VER
#define STBI_MSC_SECURE_CRT
#endif

#define STB_IMAGE_IMPLEMENTATION
#include <tinygltf/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tinygltf/stb_image_write.h>
#define TINYEXR_IMPLEMENTATION
#include <tinyexr/tinyexr.h>
#define TINYDDSLOADER_IMPLEMENTATION
#include <tinyddsloader/tinyddsloader.h>


#define CUDA_CHECK( call )                                                                                                                                                                                                                                                             \
    {                                                                                                                                                                                                                                                                                  \
        cudaError_t error = call;                                                                                                                                                                                                                                                      \
        if( error != cudaSuccess )                                                                                                                                                                                                                                                     \
            return error;                                                                                                                                                                                                                                                              \
    }

static bool fileExists( const char* path )
{
    std::ifstream str( path );
    return static_cast<bool>( str );
}

ImageBuffer loadImage( const char* fname, int32_t force_components )
{
    const std::string filename( fname );

    if( !fileExists( fname ) )
        throw std::runtime_error( ( std::string{ "sutil::loadImage(): File does not exist: " } + filename ).c_str() );

    if( filename.length() < 5 )
        throw std::runtime_error( "sutil::loadImage(): Failed to determine filename extension" );

    if( force_components > 4 || force_components == 2 || force_components == 1 )
        throw std::runtime_error( "sutil::loadImage(): Invalid force_components value" );

    ImageBuffer image;

    const std::string ext = filename.substr( filename.length() - 3 );
    if( ext == "ppm" || ext == "PPM" )
    {
        // TODO:
    }
    else if( ext == "png" || ext == "PNG" )
    {
        if( force_components != 4 && force_components != 0 )
            throw std::runtime_error( "sutil::loadImage(): PNG loading with force_components not implemented" );

        int32_t  w, h, channels;
        uint8_t* data = stbi_load( filename.c_str(), &w, &h, &channels, STBI_rgb_alpha );
        if( !data )
            throw std::runtime_error( "sutil::loadImage( png ): stbi_load failed" );

        image.create( BufferImageFormat::UNSIGNED_BYTE4, w, h, data );
        stbi_image_free( data );
    }
    else if( ext == "exr" || ext == "EXR" )
    {
        if( force_components != 4 && force_components != 0 && force_components != 3 )
            throw std::runtime_error( "sutil::loadImage(): PNG loading with force_components not implemented" );

        const char* err  = nullptr;
        float*      data = nullptr;
        int32_t     w, h;
        int32_t     res = LoadEXR( &data, &w, &h, filename.c_str(), &err );

        if( res != TINYEXR_SUCCESS )
        {
            if( err )
            {
                std::runtime_error e( ( std::string( "sutil::loadImage( exr ): " ) + err ).c_str() );
                FreeEXRErrorMessage( err );
                throw e;
            }
            else
            {
                throw std::runtime_error( "sutil::loadImage( exr ): failed to load image" );
            }
        }

        if( force_components == 4 || force_components == 0 )
        {
            image.create( BufferImageFormat::FLOAT4, w, h, data );
        }
        else  // force_components == 3
        {
            std::vector<float3> float3Data( w * h );
            for( int32_t i = 0; i < static_cast<int32_t>( w * h ); ++i )
            {
                float3Data[i].x = data[i * 4 + 0];
                float3Data[i].y = data[i * 4 + 1];
                float3Data[i].z = data[i * 4 + 2];
            }

            image.create( BufferImageFormat::FLOAT3, w, h, float3Data.data() );
        }

        free( data );
    }
    else
    {
        throw std::runtime_error( ( "sutil::loadImage(): Failed unsupported filetype '" + ext + "'" ).c_str() );
    }

    return image;
}

void savePPM( const uchar3* Pix, const char* fname, int32_t wid, int32_t hgt )
{
    if( Pix == NULL || wid < 1 || hgt < 1 )
    {
        throw std::runtime_error( "savePPM: Image is ill-formed. Not saving" );
    }

    std::ofstream OutFile( fname, std::ios::out | std::ios::binary );
    if( !OutFile.is_open() )
    {
        throw std::runtime_error( "savePPM: Could not open file for" );
    }

    OutFile << 'P';
    OutFile << '6' << std::endl;
    OutFile << wid << " " << hgt << std::endl << 255 << std::endl;

    OutFile.write( reinterpret_cast<char*>( const_cast<uchar3*>( Pix ) ), wid * hgt * sizeof( uchar3 ) );
    OutFile.close();

    return;
}

void ImageBuffer::create( BufferImageFormat pixel_format, unsigned int width, unsigned int height, void* data )
{
    destroy();

    m_pixel_format = pixel_format;
    m_width        = width;
    m_height       = height;

    switch( pixel_format )
    {
        case BufferImageFormat::UNSIGNED_BYTE4:
            m_data.uchar4Ptr = new uchar4[width * height];
            memcpy( m_data.uchar4Ptr, data, sizeof( uchar4 ) * width * height );
            break;
        case BufferImageFormat::FLOAT4:
            m_data.float4Ptr = new float4[width * height];
            memcpy( m_data.float4Ptr, data, sizeof( float4 ) * width * height );
            break;
        case BufferImageFormat::FLOAT3:
            m_data.float3Ptr = new float3[width * height];
            memcpy( m_data.float3Ptr, data, sizeof( float3 ) * width * height );
            break;
        default:
            m_data.voidPtr = 0;
    };
}

void ImageBuffer::destroy()
{
    switch( m_pixel_format )
    {
        case BufferImageFormat::UNSIGNED_BYTE4:
            delete[] m_data.uchar4Ptr;
            break;
        case BufferImageFormat::FLOAT4:
            delete[] m_data.float4Ptr;
            break;
        case BufferImageFormat::FLOAT3:
            delete[] m_data.float3Ptr;
            break;
        default:
            break;
    }

    m_pixel_format = BufferImageFormat::NONE;
    m_width        = 0u;
    m_height       = 0u;
    m_data.voidPtr = 0;
}

cudaError_t CuTexture::createFromFile( const char* fname, int32_t force_components, const struct cudaTextureDesc* overrideTexDesc, const struct cudaChannelFormatDesc* overrideDesc )
{
    cudaError_t result = {};

    ImageBuffer imageBuffer = {};

    try
    {
        std::string       filename = fname;
        const std::string ext      = filename.substr( filename.length() - 3 );
        if( ext == "dds" || ext == "DDS" )
        {
            using namespace tinyddsloader;
            DDSFile dds;
            auto    ret = dds.Load( filename.c_str() );
            if( Result::Success != ret )
            {
                if( Result::ErrorFileOpen == ret )
                    return cudaErrorFileNotFound;
                if( Result::ErrorFileOpen == ret )
                    return cudaErrorNotSupported;
                return cudaErrorUnknown;
            }

            bool layered = ( dds.GetDepth() > 1 );
            cudaExtent extent = { dds.GetWidth(), dds.GetHeight(), layered ? dds.GetDepth() : 0 };

            uint32_t blockWidth = 1;
            uint32_t bytesPerBlock = 0;

            cudaChannelFormatDesc desc;
            switch( dds.GetFormat() )
            {
                case DDSFile::DXGIFormat::B8G8R8X8_UNorm:
                case DDSFile::DXGIFormat::B8G8R8A8_UNorm:
                case DDSFile::DXGIFormat::R8G8B8A8_UNorm:
                    desc = cudaCreateChannelDesc<uchar4>();
                    break;
                case DDSFile::DXGIFormat::R32G32B32A32_UInt:
                    desc = cudaCreateChannelDesc<uint4>();
                    break;
                case DDSFile::DXGIFormat::A8_UNorm:
                    desc = cudaCreateChannelDesc<unsigned char>();
                    break;
#if CUDA_VERSION >= 11050
                case DDSFile::DXGIFormat::BC1_UNorm:
                    desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed1>();
                    blockWidth = 4;
                    bytesPerBlock = 8;
                    break;
                case DDSFile::DXGIFormat::BC2_UNorm:
                    desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed2>();
                    blockWidth = 4;
                    bytesPerBlock = 16;
                    break;
                case DDSFile::DXGIFormat::BC3_UNorm:
                    desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed3>();
                    blockWidth = 4;
                    bytesPerBlock = 16;
                    break;
                case DDSFile::DXGIFormat::BC4_UNorm:
                    desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed4>();
                    blockWidth = 4;
                    bytesPerBlock = 8;
                    break;
                case DDSFile::DXGIFormat::BC5_UNorm:
                    desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed5>();
                    blockWidth = 4;
                    bytesPerBlock = 16;
                    break;
#endif
                default:
                    return cudaErrorNotSupported;
            };
            
            if( bytesPerBlock == 0 )
            {
                // single uncompressed pixel per block
                bytesPerBlock = ( desc.x + desc.y + desc.z + desc.w ) / 8;
            }

            struct cudaResourceDesc resDesc;
            memset( &resDesc, 0, sizeof( resDesc ) );

            CUDA_CHECK( cudaMallocMipmappedArray( &m_mipmap, &desc, extent, dds.GetMipCount(), layered ? cudaArrayLayered : 0 ) );

            for( uint32_t mipIdx = 0; mipIdx < dds.GetMipCount(); mipIdx++ )
            {
                cudaArray_t levelArray;
                CUDA_CHECK( cudaGetMipmappedArrayLevel( &levelArray, m_mipmap, mipIdx ) );

                cudaExtent            levelExtent;
                cudaChannelFormatDesc levelDesc;
                CUDA_CHECK( cudaArrayGetInfo( &levelDesc, &levelExtent, 0, levelArray ) );

                const auto* imageData = dds.GetImageData( mipIdx, 0 );

                if( imageData->m_width != levelExtent.width )
                    return cudaErrorUnknown;
                if( imageData->m_height != levelExtent.height )
                    return cudaErrorUnknown;
                if( layered && imageData->m_depth != levelExtent.depth )
                    return cudaErrorUnknown;
                if( !layered && imageData->m_depth != 1 )
                    return cudaErrorUnknown;

                cudaExtent copyExtent = { levelExtent.width, levelExtent.height, 1 };
                uint32_t widthInBlocks = ( imageData->m_width + blockWidth - 1 ) / blockWidth;
                uint32_t heightInBlocks = ( imageData->m_width + blockWidth - 1 ) / blockWidth;

                cudaMemcpy3DParms copyParams = { 0 };
                copyParams.srcPos = make_cudaPos( 0, 0, 0 );
                copyParams.srcPtr = make_cudaPitchedPtr( imageData->m_mem, 
                    widthInBlocks * bytesPerBlock,
                    widthInBlocks,
                    heightInBlocks );
                copyParams.dstPos = make_cudaPos( 0, 0, 0 );
                copyParams.dstArray = levelArray;
                copyParams.extent = copyExtent;
                copyParams.kind = cudaMemcpyHostToDevice;
                CUDA_CHECK( cudaMemcpy3D( &copyParams ) );
            }

            resDesc.resType = cudaResourceTypeMipmappedArray;
            resDesc.res.mipmap.mipmap = m_mipmap;

            struct cudaTextureDesc texDesc;
            memset( &texDesc, 0, sizeof( texDesc ) );
            switch( desc.f )
            {
                case cudaChannelFormatKindFloat:
                    texDesc.readMode = cudaReadModeElementType;
                    break;
                default:
                    texDesc.readMode = cudaReadModeNormalizedFloat;
                    break;
            };
            texDesc.normalizedCoords    = 1;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.maxMipmapLevelClamp = (float)( dds.GetMipCount() - 1 );
            texDesc.mipmapFilterMode    = cudaFilterModePoint;

            CUDA_CHECK( cudaCreateTextureObject( &m_tex, &resDesc, &texDesc, 0 ) );
        }
        else
        {
            imageBuffer = loadImage( fname, force_components );

            result = create( imageBuffer, overrideTexDesc, overrideDesc );
        }
    }
    catch( ... )
    {
        result = cudaErrorUnknown;
    }

    return result;
}

bool operator!=( struct cudaChannelFormatDesc da, struct cudaChannelFormatDesc db )
{
    return ( da.f != db.f ) || ( da.x != db.x ) || ( da.y != db.y ) || ( da.z != db.z ) || ( da.w != db.w );
}

cudaError_t CuTexture::create( const ImageBuffer& image, const struct cudaTextureDesc* overrideTexDesc, const struct cudaChannelFormatDesc* overrideDesc )
{
    destroy();

    struct cudaChannelFormatDesc desc             = {};
    size_t                       pitch            = 0;
    uint32_t                     texelSizeInBytes = {};

    switch( image.format() )
    {
        break;
        case BufferImageFormat::UNSIGNED_BYTE4:
            texelSizeInBytes = 4;
            desc             = cudaCreateChannelDesc<uchar4>();
            break;
        case BufferImageFormat::FLOAT4:
            texelSizeInBytes = 16;
            desc             = cudaCreateChannelDesc<float4>();
            break;
        case BufferImageFormat::FLOAT3:
            texelSizeInBytes = 12;
            desc             = cudaCreateChannelDesc<float3>();
            break;
        default:
            return cudaErrorInvalidValue;
    };

    const void* data = image.data();

    std::vector<uint32_t> tempData;
    if( overrideDesc && ( *overrideDesc != desc ) )
    {
        // convert the input data to the requested output format
        desc = *overrideDesc;

        uint32_t texelSizeInBits = overrideDesc->x + overrideDesc->y + overrideDesc->z + overrideDesc->w;

        // texels must be byte aligned
        if( ( texelSizeInBits & 7 ) != 0 )
            return cudaErrorInvalidValue;
        texelSizeInBytes = texelSizeInBits / 8;

        switch( overrideDesc->f )
        {
            case cudaChannelFormatKindFloat:
                // only 32-bit floats are supported
                if( overrideDesc->x != 0 && overrideDesc->x != 32 )
                    return cudaErrorInvalidValue;
                if( overrideDesc->y != 0 && overrideDesc->y != 32 )
                    return cudaErrorInvalidValue;
                if( overrideDesc->z != 0 && overrideDesc->z != 32 )
                    return cudaErrorInvalidValue;
                if( overrideDesc->w != 0 && overrideDesc->w != 32 )
                    return cudaErrorInvalidValue;
                break;
            case cudaChannelFormatKindUnsigned:
            case cudaChannelFormatKindSigned:
                // texels must align with 32 bits
                if( ( ( 32 % texelSizeInBits ) % 32 ) != 0 )
                    return cudaErrorInvalidValue;
                break;
            default:
                // todo
                return cudaErrorInvalidValue;
        }

        // allocate temp memory to hold the converted texels
        tempData.resize( ( texelSizeInBytes * image.width() * image.height() + 3 ) / 4, 0 );  // round up

        for( uint32_t y = 0; y < image.height(); ++y )
            for( uint32_t x = 0; x < image.width(); ++x )
            {
                uint32_t idx   = x + y * image.width();
                float4   color = image.color( x, y );

                // convert the float4 to the output format
                switch( overrideDesc->f )
                {
                    case cudaChannelFormatKindFloat: {

                        float* dst = reinterpret_cast<float*>( reinterpret_cast<char*>( tempData.data() ) + texelSizeInBytes * idx );

                        if( overrideDesc->x )
                            *( dst++ ) = color.x;
                        if( overrideDesc->y )
                            *( dst++ ) = color.y;
                        if( overrideDesc->z )
                            *( dst++ ) = color.z;
                        if( overrideDesc->w )
                            *( dst++ ) = color.w;
                    }
                    break;
                    case cudaChannelFormatKindUnsigned: {

                        uint32_t bit = texelSizeInBits * idx;

                        const uint64_t maskX  = overrideDesc->x ? ( ( 1llu << overrideDesc->x ) - 1 ) : 0;
                        const uint32_t valueX = ( uint32_t )std::min( maskX, ( uint64_t )( color.x * maskX ) );

                        const uint64_t maskY  = overrideDesc->y ? ( ( 1llu << overrideDesc->y ) - 1 ) : 0;
                        const uint32_t valueY = ( uint32_t )std::min( maskY, ( uint64_t )( color.y * maskY ) );

                        const uint64_t maskZ  = overrideDesc->z ? ( ( 1llu << overrideDesc->z ) - 1 ) : 0;
                        const uint32_t valueZ = ( uint32_t )std::min( maskZ, ( uint64_t )( color.z * maskZ ) );

                        const uint64_t maskW  = overrideDesc->w ? ( ( 1llu << overrideDesc->w ) - 1 ) : 0;
                        const uint32_t valueW = ( uint32_t )std::min( maskW, (uint64_t)( color.w * maskW ) );

                        *( tempData.data() + ( bit / 32 ) ) |= ( valueX << ( bit & 31 ) );
                        bit += overrideDesc->x;
                        *( tempData.data() + ( bit / 32 ) ) |= ( valueY << ( bit & 31 ) );
                        bit += overrideDesc->y;
                        *( tempData.data() + ( bit / 32 ) ) |= ( valueZ << ( bit & 31 ) );
                        bit += overrideDesc->z;
                        *( tempData.data() + ( bit / 32 ) ) |= ( valueW << ( bit & 31 ) );
                        bit += overrideDesc->w;
                    }
                    break;
                    case cudaChannelFormatKindSigned: {
                        uint32_t bit = texelSizeInBits * idx;

                        const int64_t maskX = overrideDesc->x ? ( ( 1llu << ( overrideDesc->x - 1 ) ) - 1 ) : 0;
                        const uint32_t valueX = ( uint32_t )std::max( -maskX, std::min( maskX, ( int64_t )( color.x * maskX ) ) ) & ( ( ( uint32_t )maskX << 1 ) + 1 );

                        const int64_t maskY = overrideDesc->y ? ( ( 1llu << ( overrideDesc->y - 1 ) ) - 1 ) : 0;
                        const uint32_t valueY = ( uint32_t )std::max( -maskY, std::min( maskY, ( int64_t )( color.y * maskY ) ) ) & ( ( ( uint32_t )maskY << 1 ) + 1 );

                        const int64_t maskZ = overrideDesc->z ? ( ( 1llu << ( overrideDesc->z - 1 ) ) - 1 ) : 0;
                        const uint32_t valueZ = ( uint32_t )std::max( -maskZ, std::min( maskZ, ( int64_t )( color.z * maskZ ) ) ) & ( ( ( uint32_t )maskZ << 1 ) + 1 );

                        const int64_t maskW = overrideDesc->w ? ( ( 1llu << ( overrideDesc->w - 1 ) ) - 1 ) : 0;
                        const uint32_t valueW = ( uint32_t )std::max( -maskW, std::min( maskW, ( int64_t )( color.w * maskW ) ) ) & ( ( ( uint32_t )maskW << 1 ) + 1 );

                        *( tempData.data() + ( bit / 32 ) ) |= ( valueX << ( bit & 31 ) );
                        bit += overrideDesc->x;
                        *( tempData.data() + ( bit / 32 ) ) |= ( valueY << ( bit & 31 ) );
                        bit += overrideDesc->y;
                        *( tempData.data() + ( bit / 32 ) ) |= ( valueZ << ( bit & 31 ) );
                        bit += overrideDesc->z;
                        *( tempData.data() + ( bit / 32 ) ) |= ( valueW << ( bit & 31 ) );
                        bit += overrideDesc->w;
                    }
                    break;
                    default:
                        return cudaErrorInvalidValue;
                }
            }

        data = tempData.data();
    }
    CUDA_CHECK( cudaMallocPitch( (void**)&m_bitmap, &pitch, image.width() * texelSizeInBytes, image.height() ) );
    CUDA_CHECK( cudaMemcpy2D( m_bitmap, pitch, data, image.width() * texelSizeInBytes, image.width() * texelSizeInBytes, image.height(), cudaMemcpyHostToDevice ) );

    struct cudaResourceDesc resDesc;
    memset( &resDesc, 0, sizeof( resDesc ) );
    resDesc.resType                  = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr       = m_bitmap;
    resDesc.res.pitch2D.desc         = desc;
    resDesc.res.pitch2D.width        = image.width();
    resDesc.res.pitch2D.height       = image.height();
    resDesc.res.pitch2D.pitchInBytes = pitch;

    struct cudaTextureDesc texDesc;
    memset( &texDesc, 0, sizeof( texDesc ) );
    if( overrideTexDesc )
    {
        texDesc = *overrideTexDesc;
    }
    else
    {
        switch( desc.f )
        {
            case cudaChannelFormatKindFloat:
                texDesc.readMode = cudaReadModeElementType;
                break;
            default:
                texDesc.readMode = cudaReadModeNormalizedFloat;
                break;
        };
        texDesc.normalizedCoords = 1;
    }

    CUDA_CHECK( cudaCreateTextureObject( &m_tex, &resDesc, &texDesc, NULL ) );

    return cudaSuccess;
}

void CuTexture::destroy()
{
    if( m_bitmap )
        cudaFree( m_bitmap );
    if( m_tex )
        cudaDestroyTextureObject( m_tex );
    if( m_mipmap )
        cudaFreeMipmappedArray( m_mipmap );

    m_bitmap = {};
    m_tex    = {};
    m_mipmap = {};
}


ImagePPM::ImagePPM()
    : m_nx( 0u )
    , m_ny( 0u )
    , m_maxVal( 0u )
    , m_data( nullptr )
{
}

ImagePPM::ImagePPM( const std::string& filename )
    : m_nx( 0u )
    , m_ny( 0u )
    , m_maxVal( 0u )
    , m_data( nullptr )
{
    readPPM( filename );
}

ImagePPM::ImagePPM( const void* data, size_t width, size_t height, ImagePixelFormat format )
    : m_nx( 0u )
    , m_ny( 0u )
    , m_maxVal( 0u )
    , m_data( nullptr )
{
    init( data, width, height, format );
}

void ImagePPM::init( const void* data, size_t width, size_t height, ImagePixelFormat format )
{
    delete[] m_data;

    m_nx   = static_cast<unsigned int>( width );
    m_ny   = static_cast<unsigned int>( height );
    m_data = new float[width * height * 3];

    // Convert data to array of floats
    switch( format )
    {
        case IMAGE_PIXEL_FORMAT_FLOAT3: {
            const float* fdata = reinterpret_cast<const float*>( data );
            for( int j = 0; j < static_cast<int>( height ); ++j )
            {
                float*       dst = m_data + 3 * width * j;
                const float* src = fdata + width * j;
                for( unsigned int i = 0; i < width; ++i )
                {
                    // write the pixel to all 3 channels
                    *dst++ = *src;
                    *dst++ = *src;
                    *dst++ = *src++;
                }
            }

            break;
        }
        case IMAGE_PIXEL_FORMAT_FLOAT4: {
            const float* fdata = reinterpret_cast<const float*>( data );
            for( int j = 0; j < static_cast<int>( height ); ++j )
            {
                float*       dst = m_data + 3 * width * j;
                const float* src = fdata + 4 * width * j;
                for( unsigned int i = 0; i < width; ++i )
                {
                    for( int k = 0; k < 3; ++k )
                    {
                        *dst++ = *src++;
                    }
                    // skip alpha
                    ++src;
                }
            }
            break;
        }
        case IMAGE_PIXEL_FORMAT_UCHAR3: {
            const unsigned char* udata = reinterpret_cast<const unsigned char*>( data );
            for( int j = 0; j < static_cast<int>( height ); ++j )
            {
                float*               dst = m_data + 3 * width * j;
                const unsigned char* src = udata + ( 3 * width * j );
                for( unsigned int i = 0; i < width; i++ )
                {
                    *dst++ = static_cast<float>( *( src + 0 ) ) / 255.0f;
                    *dst++ = static_cast<float>( *( src + 1 ) ) / 255.0f;
                    *dst++ = static_cast<float>( *( src + 2 ) ) / 255.0f;
                    src += 3;
                }
            }
            break;
        }
        case IMAGE_PIXEL_FORMAT_UCHAR4: {
            const unsigned char* udata = reinterpret_cast<const unsigned char*>( data );
            for( int j = 0; j < static_cast<int>( height ); ++j )
            {
                float*               dst = m_data + 3 * width * j;
                const unsigned char* src = udata + ( 4 * width * j );
                for( unsigned int i = 0; i < width; i++ )
                {
                    *dst++ = static_cast<float>( *( src + 0 ) ) / 255.0f;
                    *dst++ = static_cast<float>( *( src + 1 ) ) / 255.0f;
                    *dst++ = static_cast<float>( *( src + 2 ) ) / 255.0f;
                    src += 4;  // skip alpha
                }
            }
            break;
        }
        default: {
            delete[] m_data;
            m_data = nullptr;
            std::cerr << "Image::Image( Buffer ) passed buffer with format other than IMAGE_PIXEL_FORMAT_FLOAT3, "
                         "IMAGE_PIXEL_FORMAT_FLOAT4, IMAGE_PIXEL_FORMAT_UCHAR3, or IMAGE_PIXEL_FORMAT_UCHAR4"
                      << std::endl;
        }
    }
}

void ImagePPM::compare( const ImagePPM& i0, const ImagePPM& i1, float tol, int& num_errors, float& avg_error, float& max_error )
{
    if( i0.width() != i1.width() || i0.height() != i1.height() )
    {
        throw std::string( "Image::compare passed images of differing dimensions!" );
    }
    num_errors = 0;
    max_error  = 0.0f;
    avg_error  = 0.0f;
    for( unsigned int i = 0; i < i0.width() * i0.height(); ++i )
    {
        float error[3] = {
            fabsf( i0.m_data[3 * i + 0] - i1.m_data[3 * i + 0] ),
            fabsf( i0.m_data[3 * i + 1] - i1.m_data[3 * i + 1] ),
            fabsf( i0.m_data[3 * i + 2] - i1.m_data[3 * i + 2] ),
        };
        max_error = std::max( max_error, std::max( error[0], std::max( error[1], error[2] ) ) );
        avg_error += error[0] + error[1] + error[2];
        if( error[0] > tol || error[1] > tol || error[2] > tol )
            ++num_errors;
    }
    avg_error /= static_cast<float>( i0.width() * i0.height() * 3 );
}

void ImagePPM::compare( const std::string& filename0, const std::string& filename1, float tol, int& num_errors, float& avg_error, float& max_error )
{
    ImagePPM i0( filename0 );
    ImagePPM i1( filename1 );
    if( i0.failed() )
    {
        std::stringstream ss;
        ss << "Image::compare() failed to load image file '" << filename0 << "'";
        throw ss.str();
    }
    if( i1.failed() )
    {
        std::stringstream ss;
        ss << "Image::compare() failed to load image file '" << filename1 << "'";
        throw ss.str();
    }
    compare( ImagePPM( filename0 ), ImagePPM( filename1 ), tol, num_errors, avg_error, max_error );
}

bool ImagePPM::writePPM( const std::string& filename, bool float_format )
{
    try
    {

        std::ofstream out( filename.c_str(), std::ios::out | std::ios::binary );
        if( !out )
        {
            std::cerr << "Image::writePPM failed to open outfile '" << filename << "'" << std::endl;
            return false;
        }

        if( float_format )
        {

            out << "P7\n" << m_nx << " " << m_ny << "\n" << FLT_MAX << std::endl;
            out.write( reinterpret_cast<char*>( m_data ), m_nx * m_ny * 3 * sizeof( float ) );
        }
        else
        {
            out << "P6\n" << m_nx << " " << m_ny << "\n255" << std::endl;
            for( unsigned int i = 0; i < m_nx * m_ny * 3; ++i )
            {
                float         val  = m_data[i];
                unsigned char cval = val < 0.0f ? 0u : val > 1.0f ? 255u : static_cast<unsigned char>( val * 255.0f );
                out.put( cval );
            }
        }

        return true;
    }
    catch( ... )
    {
        std::cerr << "Failed to write ppm '" << filename << "'" << std::endl;
        return false;
    }
}

ImagePPM::~ImagePPM()
{
    if( m_data )
        delete[] m_data;
}

void ImagePPM::getLine( std::ifstream& file_in, std::string& s )
{
    for( ;; )
    {
        if( !std::getline( file_in, s ) )
            return;
        std::string::size_type index = s.find_first_not_of( "\n\r\t " );
        if( index != std::string::npos && s[index] != '#' )
            break;
    }
}

void ImagePPM::readPPM( const std::string& filename )
{
    delete[] m_data;
    m_data = nullptr;

    if( filename.empty() )
        return;

    // Open file
    try
    {
        std::ifstream file_in( filename.c_str(), std::ifstream::in | std::ifstream::binary );
        if( !file_in )
        {
            std::cerr << "Image( '" << filename << "' ) failed to open file." << std::endl;
            return;
        }

        // Check magic number to make sure we have an ascii or binary PPM
        std::string line, magic_number;
        getLine( file_in, line );
        std::istringstream iss1( line );
        iss1 >> magic_number;
        if( magic_number != "P6" && magic_number != "P3" && magic_number != "P7" )
        {
            std::cerr << "Image( '" << filename << "' ) unknown magic number: " << magic_number << ".  Only P3, P6 and P7 supported." << std::endl;
            return;
        }

        // width, height
        getLine( file_in, line );
        std::istringstream iss2( line );
        iss2 >> m_nx >> m_ny;

        // max channel value
        getLine( file_in, line );
        std::istringstream iss3( line );
        iss3 >> m_maxVal;

        m_data = new float[m_nx * m_ny * 3];

        if( magic_number == "P3" )
        {
            unsigned int num_elements = m_nx * m_ny * 3;
            unsigned int count        = 0;

            while( count < num_elements )
            {
                getLine( file_in, line );
                std::istringstream iss( line );

                while( iss.good() )
                {
                    unsigned int c;
                    iss >> c;
                    m_data[count++] = static_cast<float>( c ) / 255.0f;
                }
            }
        }
        else if( magic_number == "P6" )
        {

            unsigned char* charm_data = new unsigned char[m_nx * m_ny * 3];
            file_in.read( reinterpret_cast<char*>( charm_data ), m_nx * m_ny * 3 );
            for( unsigned int i = 0u; i < m_nx * m_ny * 3; ++i )
            {
                m_data[i] = static_cast<float>( charm_data[i] ) / 255.0f;
            }
            delete[] charm_data;
        }
        else if( magic_number == "P7" )
        {

            file_in.read( reinterpret_cast<char*>( m_data ), m_nx * m_ny * 3 * sizeof( float ) );
        }
    }
    catch( ... )
    {
        std::cerr << "Image( '" << filename << "' ) failed to load" << std::endl;
        if( m_data )
            delete[] m_data;
        m_data = nullptr;
    }
}
