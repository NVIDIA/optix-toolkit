#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

#include <iostream>

#define TINYDDSLOADER_IMPLEMENTATION
#include "tinyddsloader.h"

#ifndef GL_EXT_texture_compression_s3tc
#define GL_EXT_texture_compression_s3tc 1
#define GL_COMPRESSED_RGB_S3TC_DXT1_EXT 0x83F0
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT 0x83F1
#define GL_COMPRESSED_RGBA_S3TC_DXT3_EXT 0x83F2
#define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT 0x83F3
#define GL_COMPRESSED_RED_RGTC1_EXT 0x8DBB
#define GL_COMPRESSED_SIGNED_RED_RGTC1_EXT 0x8DBC
#define GL_COMPRESSED_RED_GREEN_RGTC2_EXT 0x8DBD
#define GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT 0x8DBE
#endif /* GL_EXT_texture_compression_s3tc */

using namespace tinyddsloader;

struct GLSwizzle {
    GLenum m_r, m_g, m_b, m_a;
};

struct GLFormat {
    DDSFile::DXGIFormat m_dxgiFormat;
    GLenum m_type;
    GLenum m_format;
    GLSwizzle m_swizzle;
};

bool TranslateFormat(DDSFile::DXGIFormat fmt, GLFormat* outFormat) {
    static const GLSwizzle sws[] = {
        {GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA},
        {GL_BLUE, GL_GREEN, GL_RED, GL_ALPHA},
        {GL_BLUE, GL_GREEN, GL_RED, GL_ONE},
        {GL_RED, GL_GREEN, GL_BLUE, GL_ONE},
        {GL_RED, GL_ZERO, GL_ZERO, GL_ZERO},
        {GL_RED, GL_GREEN, GL_ZERO, GL_ZERO},
    };
    using DXGIFmt = DDSFile::DXGIFormat;
    static const GLFormat formats[] = {
        {DXGIFmt::R8G8B8A8_UNorm, GL_UNSIGNED_BYTE, GL_RGBA, sws[0]},
        {DXGIFmt::B8G8R8A8_UNorm, GL_UNSIGNED_BYTE, GL_RGBA, sws[1]},
        {DXGIFmt::B8G8R8X8_UNorm, GL_UNSIGNED_BYTE, GL_RGBA, sws[2]},
        {DXGIFmt::BC1_UNorm, 0, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, sws[0]},
        {DXGIFmt::BC2_UNorm, 0, GL_COMPRESSED_RGBA_S3TC_DXT3_EXT, sws[0]},
        {DXGIFmt::BC3_UNorm, 0, GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, sws[0]},
        {DXGIFmt::BC4_UNorm, 0, GL_COMPRESSED_RED_RGTC1_EXT, sws[0]},
        {DXGIFmt::BC4_SNorm, 0, GL_COMPRESSED_SIGNED_RED_RGTC1_EXT, sws[0]},
        {DXGIFmt::BC5_UNorm, 0, GL_COMPRESSED_RED_GREEN_RGTC2_EXT, sws[0]},
        {DXGIFmt::BC5_SNorm, 0, GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT,
         sws[0]},
    };
    for (const auto& format : formats) {
        if (format.m_dxgiFormat == fmt) {
            if (outFormat) {
                *outFormat = format;
            }
            return true;
        }
    }
    return false;
}

bool IsCompressed(GLenum fmt) {
    switch (fmt) {
        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
        case GL_COMPRESSED_RED_RGTC1_EXT:
        case GL_COMPRESSED_SIGNED_RED_RGTC1_EXT:
        case GL_COMPRESSED_RED_GREEN_RGTC2_EXT:
        case GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT:
            return true;
        default:
            return false;
    }
}

bool LoadGLTexture(GLuint tex, DDSFile& dds) {
    GLenum target = GL_INVALID_ENUM;
    bool isArray = false;
    if (dds.GetTextureDimension() == DDSFile::TextureDimension::Texture1D) {
        if (dds.GetArraySize() > 1) {
            target = GL_TEXTURE_1D_ARRAY;
            isArray = true;
        } else {
            target = GL_TEXTURE_1D;
        }
    } else if (dds.GetTextureDimension() ==
               DDSFile::TextureDimension::Texture2D) {
        if (dds.GetArraySize() > 1) {
            if (dds.IsCubemap()) {
                if (dds.GetArraySize() > 6) {
                    target = GL_TEXTURE_CUBE_MAP_ARRAY;
                    isArray = true;
                } else {
                    target = GL_TEXTURE_CUBE_MAP;
                }
            } else {
                target = GL_TEXTURE_2D_ARRAY;
                isArray = true;
            }
        } else {
            target = GL_TEXTURE_2D;
        }
    } else if (dds.GetTextureDimension() ==
               DDSFile::TextureDimension::Texture3D) {
        target = GL_TEXTURE_3D;
    }

    GLFormat format;
    if (!TranslateFormat(dds.GetFormat(), &format)) {
        return false;
    }

    glBindTexture(target, tex);
    glTexParameteri(target, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(target, GL_TEXTURE_MAX_LEVEL, dds.GetMipCount() - 1);
    glTexParameteri(target, GL_TEXTURE_SWIZZLE_R, format.m_swizzle.m_r);
    glTexParameteri(target, GL_TEXTURE_SWIZZLE_G, format.m_swizzle.m_g);
    glTexParameteri(target, GL_TEXTURE_SWIZZLE_B, format.m_swizzle.m_b);
    glTexParameteri(target, GL_TEXTURE_SWIZZLE_A, format.m_swizzle.m_a);

    switch (target) {
        case GL_TEXTURE_1D:
            glTexStorage1D(target, dds.GetMipCount(), format.m_format,
                           dds.GetWidth());
            break;
        case GL_TEXTURE_1D_ARRAY:
            glTexStorage2D(target, dds.GetMipCount(), format.m_format,
                           dds.GetWidth(), dds.GetArraySize());
            break;
        case GL_TEXTURE_2D:
            glTexStorage2D(target, dds.GetMipCount(), format.m_format,
                           dds.GetWidth(), dds.GetHeight());
            break;
        case GL_TEXTURE_CUBE_MAP:
            glTexStorage2D(target, dds.GetMipCount(), format.m_format,
                           dds.GetWidth(), dds.GetHeight());
            break;
        case GL_TEXTURE_2D_ARRAY:
            glTexStorage3D(target, dds.GetMipCount(), format.m_format,
                           dds.GetWidth(), dds.GetHeight(), dds.GetArraySize());
            break;
        case GL_TEXTURE_3D:
            glTexStorage3D(target, dds.GetMipCount(), format.m_format,
                           dds.GetWidth(), dds.GetHeight(), dds.GetDepth());
            break;
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            glTexStorage3D(target, dds.GetMipCount(), format.m_format,
                           dds.GetWidth(), dds.GetHeight(), dds.GetArraySize());
            break;
        default:
            glBindTexture(target, 0);
            return false;
    }
    dds.Flip();

    uint32_t numFaces = dds.IsCubemap() ? 6 : 1;
    for (uint32_t layer = 0; layer < dds.GetArraySize(); layer++) {
        for (uint32_t face = 0; face < numFaces; face++) {
            for (uint32_t level = 0; level < dds.GetMipCount(); level++) {
                GLenum target2 = dds.IsCubemap()
                                     ? (GL_TEXTURE_CUBE_MAP_POSITIVE_X + face)
                                     : target;
                auto imageData = dds.GetImageData(level, layer * numFaces);
                switch (target) {
                    case GL_TEXTURE_1D:
                        if (IsCompressed(format.m_format)) {
                            glCompressedTexSubImage1D(
                                target2, level, 0, imageData->m_width,
                                format.m_format, imageData->m_memSlicePitch,
                                imageData->m_mem);
                        } else {
                            glTexSubImage1D(target2, level, 0,
                                            imageData->m_width, format.m_format,
                                            format.m_type, imageData->m_mem);
                        }
                        break;
                    case GL_TEXTURE_1D_ARRAY:
                    case GL_TEXTURE_2D:
                    case GL_TEXTURE_CUBE_MAP: {
                        auto w = imageData->m_width;
                        auto h = isArray ? layer : imageData->m_height;
                        if (IsCompressed(format.m_format)) {
                            glCompressedTexSubImage2D(
                                target2, level, 0, 0, w, h, format.m_format,
                                imageData->m_memSlicePitch, imageData->m_mem);
                        } else {
                            glTexSubImage2D(target2, level, 0, 0, w, h,
                                            format.m_format, format.m_type,
                                            imageData->m_mem);
                        }
                        break;
                    }
                    default:
                        glBindTexture(target, 0);
                        return false;
                }
            }
        }
    }

    glBindTexture(target, 0);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        return 1;
    }

    DDSFile dds;
    auto ret = dds.Load(argv[1]);
    if (tinyddsloader::Result::Success != ret) {
        std::cout << "Failed to load.[" << argv[1] << "]\n";
        std::cout << "Result : " << int(ret) << "\n";
        return 1;
    }

    if (!glfwInit()) {
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = nullptr;
    window = glfwCreateWindow(800, 600, "ddsloader", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);

    if (gl3wInit() != 0) {
        return 1;
    }

    GLuint tex;
    glGenTextures(1, &tex);

    if (!LoadGLTexture(tex, dds)) {
        return 1;
    }

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}
