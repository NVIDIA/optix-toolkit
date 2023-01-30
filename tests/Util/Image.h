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

#include <algorithm>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string>

enum class BufferImageFormat
{
    NONE,
    UNSIGNED_BYTE4,
    FLOAT4,
    FLOAT3
};

// Helper class for 2d image data management and format independant read access
struct ImageBuffer
{
  public:
    // image buffer is only movable
    ImageBuffer() = default;
    ImageBuffer( ImageBuffer&& source ) noexcept { swap( source ); };
    ImageBuffer( const ImageBuffer& ) = delete;
    ImageBuffer& operator             =( ImageBuffer&& source ) noexcept
    {
        swap( source );
        return *this;
    };
    ImageBuffer& operator=( const ImageBuffer& ) = delete;

    ~ImageBuffer() { destroy(); }

    void create( BufferImageFormat pixel_format, unsigned int width, unsigned int height, void* data );

    unsigned int width() const { return m_width; }

    unsigned int height() const { return m_height; }

    BufferImageFormat format() const { return m_pixel_format; }

    // format independent load of a pixel in the image
    inline float4 color( unsigned int x, unsigned int y ) const
    {
        uint32_t idx = x + y * m_width;

        float4 color = {};
        // convert the input color to float4
        switch( m_pixel_format )
        {
            case BufferImageFormat::UNSIGNED_BYTE4: {
                uchar4 inColor = m_data.uchar4Ptr[idx];
                color          = { inColor.x / 255.f, inColor.y / 255.f, inColor.z / 255.f, inColor.w / 255.f };
            }
            break;
            case BufferImageFormat::FLOAT4: {
                color = m_data.float4Ptr[idx];
            }
            break;
            case BufferImageFormat::FLOAT3: {
                float3 inColor = m_data.float3Ptr[idx];
                color          = float4{ inColor.x, inColor.y, inColor.z, 1.f };
            }
            break;
            default:
                break;
        }

        return color;
    }

    // raw data pointer
    inline const void* data() const
    {
        // this switch should collapse as all cases are identical, but reading m_data.voidPtr after writing any of the others is undefined.
        switch( m_pixel_format )
        {
            case BufferImageFormat::UNSIGNED_BYTE4:
                return (const void*)m_data.uchar4Ptr;
            case BufferImageFormat::FLOAT4:
                return (const void*)m_data.float4Ptr;
            case BufferImageFormat::FLOAT3:
                return (const void*)m_data.float3Ptr;
            default:
                break;
        }
        return m_data.voidPtr;
    }

  private:
    void swap( ImageBuffer& source )
    {
        std::swap( source.m_pixel_format, m_pixel_format );
        std::swap( source.m_width, m_width );
        std::swap( source.m_height, m_height );
        std::swap( source.m_data, m_data );
    }

    void destroy();

    union
    {
        float4* float4Ptr;
        float3* float3Ptr;
        uchar4* uchar4Ptr;
        void*   voidPtr;
    } m_data = {};

    unsigned int      m_width        = 0u;
    unsigned int      m_height       = 0u;
    BufferImageFormat m_pixel_format = BufferImageFormat::NONE;
};

void savePPM( const uchar3* Pix, const char* fname, int32_t wid, int32_t hgt );

ImageBuffer loadImage( const char* fname, int32_t force_components );

// Helper class for construction and management of cuda texture data
class CuTexture
{
  public:
    CuTexture(){};
    ~CuTexture() { destroy(); };

    CuTexture( const CuTexture& source ) = delete;
    CuTexture( CuTexture&& source ) noexcept { swap( source ); }

    CuTexture& operator=( const CuTexture& source ) = delete;

    CuTexture& operator=( CuTexture&& source ) noexcept
    {
        swap( source );
        return *this;
    }

    // create a plain cuda texture from an image buffer
    cudaError_t create( const ImageBuffer& image, const struct cudaTextureDesc* overrideTexDesc = 0, const struct cudaChannelFormatDesc* overrideDesc = 0 );

    // create a plain cuda texture from a file
    cudaError_t createFromFile( const char* fname, int32_t force_components = 0, const struct cudaTextureDesc* overrideTexDesc = 0, const struct cudaChannelFormatDesc* overrideDesc = 0 );

    cudaTextureObject_t getTexture() const { return m_tex; }

    void destroy();

  private:
    void swap( CuTexture& source ) noexcept
    {
        std::swap( m_tex, source.m_tex );
        std::swap( m_bitmap, source.m_bitmap );
        std::swap( m_mipmap, source.m_mipmap );
    }

    cudaTextureObject_t  m_tex    = {};
    uint8_t*             m_bitmap = {};
    cudaMipmappedArray_t m_mipmap = {};
};

typedef enum ImagePixelFormat
{
    IMAGE_PIXEL_FORMAT_FLOAT3 = 1,  ///< three floats, RGB
    IMAGE_PIXEL_FORMAT_FLOAT4 = 2,  ///< four floats, RGBA
    IMAGE_PIXEL_FORMAT_UCHAR3 = 3,  ///< four unsigned chars, RGB
    IMAGE_PIXEL_FORMAT_UCHAR4 = 4,  ///< four unsigned chars, RGBA
} ImagePixelFormat;

// Helper class for loading and data management of PPM image files.
class ImagePPM
{
  public:
    ImagePPM();

    // Initialize image from PPM file
    explicit ImagePPM( const std::string& filename );

    // Initialize image from buffer
    ImagePPM( const void* data, size_t width, size_t height, ImagePixelFormat format );

    ~ImagePPM();

    void init( const void* data, size_t width, size_t height, ImagePixelFormat format );

    void readPPM( const std::string& filename );

    // Store image object to disk in PPM format (or PPM-like raw float format)
    bool writePPM( const std::string& filename, bool float_format = true );

    bool         failed() const { return m_data == nullptr; }
    unsigned int width() const { return m_nx; }
    unsigned int height() const { return m_ny; }
    float*       raster() const { return m_data; }

    static void compare( const ImagePPM& i0, const ImagePPM& i1, float tol, int& num_errors, float& avg_error, float& max_error );
    static void compare( const std::string& filename0, const std::string& filename1, float tol, int& num_errors, float& avg_error, float& max_error );

  private:
    unsigned int m_nx;
    unsigned int m_ny;
    unsigned int m_maxVal;
    float*       m_data;  // r,g,b triples

    ImagePPM( const ImagePPM& );  // forbidden
    static void getLine( std::ifstream& file_in, std::string& s );
};
