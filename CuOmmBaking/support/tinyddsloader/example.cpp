#include <iostream>

#define TINYDDSLOADER_IMPLEMENTATION
#include "tinyddsloader.h"

using namespace tinyddsloader;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "example <dds file>\n";
        return 1;
    }

    DDSFile dds;
    auto ret = dds.Load(argv[1]);
    if (tinyddsloader::Result::Success != ret) {
        std::cout << "Failed to load.[" << argv[1] << "]\n";
        std::cout << "Result : " << int(ret) << "\n";
        return 1;
    }

    std::cout << "Width: " << dds.GetWidth() << "\n";
    std::cout << "Height: " << dds.GetHeight() << "\n";
    std::cout << "Depth: " << dds.GetDepth() << "\n";

    std::cout << "Mip: " << dds.GetMipCount() << "\n";
    std::cout << "Array: " << dds.GetArraySize() << "\n";
    for (uint32_t arrayIdx = 0; arrayIdx < dds.GetArraySize(); arrayIdx++) {
        for (uint32_t mipIdx = 0; mipIdx < dds.GetMipCount(); mipIdx++) {
            const auto* imageData = dds.GetImageData(mipIdx, arrayIdx);
            std::cout << "Array[" << arrayIdx << "] "
                      << "Mip[" << mipIdx << "]: "
                      << "(" << imageData->m_width << ", "
                      << imageData->m_height << ", " << imageData->m_depth
                      << ")\n";
        }
    }
    std::cout << "Cubemap: " << dds.IsCubemap() << "\n";

    std::cout << "Dimension: ";
    switch (dds.GetTextureDimension()) {
        case DDSFile::TextureDimension::Texture1D:
            std::cout << "Texture1D";
            break;
        case DDSFile::TextureDimension::Texture2D:
            std::cout << "Texture2D";
            break;
        case DDSFile::TextureDimension::Texture3D:
            std::cout << "Texture3D";
            break;
        default:
            std::cout << "Unknown";
            break;
    }
    std::cout << "\n";

    std::cout << "Format: ";
    switch (dds.GetFormat()) {
        case DDSFile::DXGIFormat::R32G32B32A32_Typeless:
            std::cout << "R32G32B32A32_Typeless";
            break;
        case DDSFile::DXGIFormat::R32G32B32A32_Float:
            std::cout << "R32G32B32A32_Float";
            break;
        case DDSFile::DXGIFormat::R32G32B32A32_UInt:
            std::cout << "R32G32B32A32_UInt";
            break;
        case DDSFile::DXGIFormat::R32G32B32A32_SInt:
            std::cout << "R32G32B32A32_SInt";
            break;
        case DDSFile::DXGIFormat::R32G32B32_Typeless:
            std::cout << "R32G32B32_Typeless";
            break;
        case DDSFile::DXGIFormat::R32G32B32_Float:
            std::cout << "R32G32B32_Float";
            break;
        case DDSFile::DXGIFormat::R32G32B32_UInt:
            std::cout << "R32G32B32_UInt";
            break;
        case DDSFile::DXGIFormat::R32G32B32_SInt:
            std::cout << "R32G32B32_SInt";
            break;
        case DDSFile::DXGIFormat::R16G16B16A16_Typeless:
            std::cout << "R16G16B16A16_Typeless";
            break;
        case DDSFile::DXGIFormat::R16G16B16A16_Float:
            std::cout << "R16G16B16A16_Float";
            break;
        case DDSFile::DXGIFormat::R16G16B16A16_UNorm:
            std::cout << "R16G16B16A16_UNorm";
            break;
        case DDSFile::DXGIFormat::R16G16B16A16_UInt:
            std::cout << "R16G16B16A16_UInt";
            break;
        case DDSFile::DXGIFormat::R16G16B16A16_SNorm:
            std::cout << "R16G16B16A16_SNorm";
            break;
        case DDSFile::DXGIFormat::R16G16B16A16_SInt:
            std::cout << "R16G16B16A16_SInt";
            break;
        case DDSFile::DXGIFormat::R32G32_Typeless:
            std::cout << "R32G32_Typeless";
            break;
        case DDSFile::DXGIFormat::R32G32_Float:
            std::cout << "R32G32_Float";
            break;
        case DDSFile::DXGIFormat::R32G32_UInt:
            std::cout << "R32G32_UInt";
            break;
        case DDSFile::DXGIFormat::R32G32_SInt:
            std::cout << "R32G32_SInt";
            break;
        case DDSFile::DXGIFormat::R32G8X24_Typeless:
            std::cout << "R32G8X24_Typeless";
            break;
        case DDSFile::DXGIFormat::D32_Float_S8X24_UInt:
            std::cout << "D32_Float_S8X24_UInt";
            break;
        case DDSFile::DXGIFormat::R32_Float_X8X24_Typeless:
            std::cout << "R32_Float_X8X24_Typeless";
            break;
        case DDSFile::DXGIFormat::X32_Typeless_G8X24_UInt:
            std::cout << "X32_Typeless_G8X24_UInt";
            break;
        case DDSFile::DXGIFormat::R10G10B10A2_Typeless:
            std::cout << "R10G10B10A2_Typeless";
            break;
        case DDSFile::DXGIFormat::R10G10B10A2_UNorm:
            std::cout << "R10G10B10A2_UNorm";
            break;
        case DDSFile::DXGIFormat::R10G10B10A2_UInt:
            std::cout << "R10G10B10A2_UInt";
            break;
        case DDSFile::DXGIFormat::R11G11B10_Float:
            std::cout << "R11G11B10_Float";
            break;
        case DDSFile::DXGIFormat::R8G8B8A8_Typeless:
            std::cout << "R8G8B8A8_Typeless";
            break;
        case DDSFile::DXGIFormat::R8G8B8A8_UNorm:
            std::cout << "R8G8B8A8_UNorm";
            break;
        case DDSFile::DXGIFormat::R8G8B8A8_UNorm_SRGB:
            std::cout << "R8G8B8A8_UNorm_SRGB";
            break;
        case DDSFile::DXGIFormat::R8G8B8A8_UInt:
            std::cout << "R8G8B8A8_UInt";
            break;
        case DDSFile::DXGIFormat::R8G8B8A8_SNorm:
            std::cout << "R8G8B8A8_SNorm";
            break;
        case DDSFile::DXGIFormat::R8G8B8A8_SInt:
            std::cout << "R8G8B8A8_SInt";
            break;
        case DDSFile::DXGIFormat::R16G16_Typeless:
            std::cout << "R16G16_Typeless";
            break;
        case DDSFile::DXGIFormat::R16G16_Float:
            std::cout << "R16G16_Float";
            break;
        case DDSFile::DXGIFormat::R16G16_UNorm:
            std::cout << "R16G16_UNorm";
            break;
        case DDSFile::DXGIFormat::R16G16_UInt:
            std::cout << "R16G16_UInt";
            break;
        case DDSFile::DXGIFormat::R16G16_SNorm:
            std::cout << "R16G16_SNorm";
            break;
        case DDSFile::DXGIFormat::R16G16_SInt:
            std::cout << "R16G16_SInt";
            break;
        case DDSFile::DXGIFormat::R32_Typeless:
            std::cout << "R32_Typeless";
            break;
        case DDSFile::DXGIFormat::D32_Float:
            std::cout << "D32_Float";
            break;
        case DDSFile::DXGIFormat::R32_Float:
            std::cout << "R32_Float";
            break;
        case DDSFile::DXGIFormat::R32_UInt:
            std::cout << "R32_UInt";
            break;
        case DDSFile::DXGIFormat::R32_SInt:
            std::cout << "R32_SInt";
            break;
        case DDSFile::DXGIFormat::R24G8_Typeless:
            std::cout << "R24G8_Typeless";
            break;
        case DDSFile::DXGIFormat::D24_UNorm_S8_UInt:
            std::cout << "D24_UNorm_S8_UInt";
            break;
        case DDSFile::DXGIFormat::R24_UNorm_X8_Typeless:
            std::cout << "R24_UNorm_X8_Typeless";
            break;
        case DDSFile::DXGIFormat::X24_Typeless_G8_UInt:
            std::cout << "X24_Typeless_G8_UInt";
            break;
        case DDSFile::DXGIFormat::R8G8_Typeless:
            std::cout << "R8G8_Typeless";
            break;
        case DDSFile::DXGIFormat::R8G8_UNorm:
            std::cout << "R8G8_UNorm";
            break;
        case DDSFile::DXGIFormat::R8G8_UInt:
            std::cout << "R8G8_UInt";
            break;
        case DDSFile::DXGIFormat::R8G8_SNorm:
            std::cout << "R8G8_SNorm";
            break;
        case DDSFile::DXGIFormat::R8G8_SInt:
            std::cout << "R8G8_SInt";
            break;
        case DDSFile::DXGIFormat::R16_Typeless:
            std::cout << "R16_Typeless";
            break;
        case DDSFile::DXGIFormat::R16_Float:
            std::cout << "R16_Float";
            break;
        case DDSFile::DXGIFormat::D16_UNorm:
            std::cout << "D16_UNorm";
            break;
        case DDSFile::DXGIFormat::R16_UNorm:
            std::cout << "R16_UNorm";
            break;
        case DDSFile::DXGIFormat::R16_UInt:
            std::cout << "R16_UInt";
            break;
        case DDSFile::DXGIFormat::R16_SNorm:
            std::cout << "R16_SNorm";
            break;
        case DDSFile::DXGIFormat::R16_SInt:
            std::cout << "R16_SInt";
            break;
        case DDSFile::DXGIFormat::R8_Typeless:
            std::cout << "R8_Typeless";
            break;
        case DDSFile::DXGIFormat::R8_UNorm:
            std::cout << "R8_UNorm";
            break;
        case DDSFile::DXGIFormat::R8_UInt:
            std::cout << "R8_UInt";
            break;
        case DDSFile::DXGIFormat::R8_SNorm:
            std::cout << "R8_SNorm";
            break;
        case DDSFile::DXGIFormat::R8_SInt:
            std::cout << "R8_SInt";
            break;
        case DDSFile::DXGIFormat::A8_UNorm:
            std::cout << "A8_UNorm";
            break;
        case DDSFile::DXGIFormat::R1_UNorm:
            std::cout << "R1_UNorm";
            break;
        case DDSFile::DXGIFormat::R9G9B9E5_SHAREDEXP:
            std::cout << "R9G9B9E5_SHAREDEXP";
            break;
        case DDSFile::DXGIFormat::R8G8_B8G8_UNorm:
            std::cout << "R8G8_B8G8_UNorm";
            break;
        case DDSFile::DXGIFormat::G8R8_G8B8_UNorm:
            std::cout << "G8R8_G8B8_UNorm";
            break;
        case DDSFile::DXGIFormat::BC1_Typeless:
            std::cout << "BC1_Typeless";
            break;
        case DDSFile::DXGIFormat::BC1_UNorm:
            std::cout << "BC1_UNorm";
            break;
        case DDSFile::DXGIFormat::BC1_UNorm_SRGB:
            std::cout << "BC1_UNorm_SRGB";
            break;
        case DDSFile::DXGIFormat::BC2_Typeless:
            std::cout << "BC2_Typeless";
            break;
        case DDSFile::DXGIFormat::BC2_UNorm:
            std::cout << "BC2_UNorm";
            break;
        case DDSFile::DXGIFormat::BC2_UNorm_SRGB:
            std::cout << "BC2_UNorm_SRGB";
            break;
        case DDSFile::DXGIFormat::BC3_Typeless:
            std::cout << "BC3_Typeless";
            break;
        case DDSFile::DXGIFormat::BC3_UNorm:
            std::cout << "BC3_UNorm";
            break;
        case DDSFile::DXGIFormat::BC3_UNorm_SRGB:
            std::cout << "BC3_UNorm_SRGB";
            break;
        case DDSFile::DXGIFormat::BC4_Typeless:
            std::cout << "BC4_Typeless";
            break;
        case DDSFile::DXGIFormat::BC4_UNorm:
            std::cout << "BC4_UNorm";
            break;
        case DDSFile::DXGIFormat::BC4_SNorm:
            std::cout << "BC4_SNorm";
            break;
        case DDSFile::DXGIFormat::BC5_Typeless:
            std::cout << "BC5_Typeless";
            break;
        case DDSFile::DXGIFormat::BC5_UNorm:
            std::cout << "BC5_UNorm";
            break;
        case DDSFile::DXGIFormat::BC5_SNorm:
            std::cout << "BC5_SNorm";
            break;
        case DDSFile::DXGIFormat::B5G6R5_UNorm:
            std::cout << "B5G6R5_UNorm";
            break;
        case DDSFile::DXGIFormat::B5G5R5A1_UNorm:
            std::cout << "B5G5R5A1_UNorm";
            break;
        case DDSFile::DXGIFormat::B8G8R8A8_UNorm:
            std::cout << "B8G8R8A8_UNorm";
            break;
        case DDSFile::DXGIFormat::B8G8R8X8_UNorm:
            std::cout << "B8G8R8X8_UNorm";
            break;
        case DDSFile::DXGIFormat::R10G10B10_XR_BIAS_A2_UNorm:
            std::cout << "R10G10B10_XR_BIAS_A2_UNorm";
            break;
        case DDSFile::DXGIFormat::B8G8R8A8_Typeless:
            std::cout << "B8G8R8A8_Typeless";
            break;
        case DDSFile::DXGIFormat::B8G8R8A8_UNorm_SRGB:
            std::cout << "B8G8R8A8_UNorm_SRGB";
            break;
        case DDSFile::DXGIFormat::B8G8R8X8_Typeless:
            std::cout << "B8G8R8X8_Typeless";
            break;
        case DDSFile::DXGIFormat::B8G8R8X8_UNorm_SRGB:
            std::cout << "B8G8R8X8_UNorm_SRGB";
            break;
        case DDSFile::DXGIFormat::BC6H_Typeless:
            std::cout << "BC6H_Typeless";
            break;
        case DDSFile::DXGIFormat::BC6H_UF16:
            std::cout << "BC6H_UF16";
            break;
        case DDSFile::DXGIFormat::BC6H_SF16:
            std::cout << "BC6H_SF16";
            break;
        case DDSFile::DXGIFormat::BC7_Typeless:
            std::cout << "BC7_Typeless";
            break;
        case DDSFile::DXGIFormat::BC7_UNorm:
            std::cout << "BC7_UNorm";
            break;
        case DDSFile::DXGIFormat::BC7_UNorm_SRGB:
            std::cout << "BC7_UNorm_SRGB";
            break;
        case DDSFile::DXGIFormat::AYUV:
            std::cout << "AYUV";
            break;
        case DDSFile::DXGIFormat::Y410:
            std::cout << "Y410";
            break;
        case DDSFile::DXGIFormat::Y416:
            std::cout << "Y416";
            break;
        case DDSFile::DXGIFormat::NV12:
            std::cout << "NV12";
            break;
        case DDSFile::DXGIFormat::P010:
            std::cout << "P010";
            break;
        case DDSFile::DXGIFormat::P016:
            std::cout << "P016";
            break;
        case DDSFile::DXGIFormat::YUV420_OPAQUE:
            std::cout << "YUV420_OPAQUE";
            break;
        case DDSFile::DXGIFormat::YUY2:
            std::cout << "YUY2";
            break;
        case DDSFile::DXGIFormat::Y210:
            std::cout << "Y210";
            break;
        case DDSFile::DXGIFormat::Y216:
            std::cout << "Y216";
            break;
        case DDSFile::DXGIFormat::NV11:
            std::cout << "NV11";
            break;
        case DDSFile::DXGIFormat::AI44:
            std::cout << "AI44";
            break;
        case DDSFile::DXGIFormat::IA44:
            std::cout << "IA44";
            break;
        case DDSFile::DXGIFormat::P8:
            std::cout << "P8";
            break;
        case DDSFile::DXGIFormat::A8P8:
            std::cout << "A8P8";
            break;
        case DDSFile::DXGIFormat::B4G4R4A4_UNorm:
            std::cout << "B4G4R4A4_UNorm";
            break;
        case DDSFile::DXGIFormat::P208:
            std::cout << "P208";
            break;
        case DDSFile::DXGIFormat::V208:
            std::cout << "V208";
            break;
        case DDSFile::DXGIFormat::V408:
            std::cout << "V408";
            break;
    }
    std::cout << "\n";

    return 0;
}
