// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>
#include <OptiXToolkit/PbrtSceneLoader/SceneDescription.h>

#include <vector_types.h>

#include <memory>
#include <optional>
#include <string>

namespace demandPbrtScene {

constexpr unsigned int PBRT_CHECKERBOARD_TEXTURE_SIZE{ 1024U };

struct PbrtCheckerboardDefinition
{
    float4 tex1;
    float4 tex2;
    float  uscale;
    float  vscale;
    float  udelta;
    float  vdelta;
};

inline bool operator==( const PbrtCheckerboardDefinition& lhs, const PbrtCheckerboardDefinition& rhs )
{
    return lhs.tex1.x == rhs.tex1.x && lhs.tex1.y == rhs.tex1.y && lhs.tex1.z == rhs.tex1.z
           && lhs.tex1.w == rhs.tex1.w && lhs.tex2.x == rhs.tex2.x && lhs.tex2.y == rhs.tex2.y
           && lhs.tex2.z == rhs.tex2.z && lhs.tex2.w == rhs.tex2.w && lhs.uscale == rhs.uscale
           && lhs.vscale == rhs.vscale && lhs.udelta == rhs.udelta && lhs.vdelta == rhs.vdelta;
}

std::string makePbrtCheckerboardTextureKey( const PbrtCheckerboardDefinition& definition );

bool isPbrtCheckerboardTextureKey( const std::string& key );

PbrtCheckerboardDefinition parsePbrtCheckerboardTextureKey( const std::string& key );

std::optional<PbrtCheckerboardDefinition> pbrtCheckerboardDefinition( const otk::pbrt::PbrtTexture& texture );

std::string pbrtCheckerboardTextureKey( const otk::pbrt::PbrtTexture& texture );

std::shared_ptr<imageSource::ImageSource> createPbrtCheckerboardImageSource(
    const PbrtCheckerboardDefinition& definition );

std::shared_ptr<imageSource::ImageSource> createPbrtCheckerboardImageSource( const std::string& key );

class PbrtCheckerboardImageSource : public imageSource::ImageSourceBase
{
  public:
    explicit PbrtCheckerboardImageSource( PbrtCheckerboardDefinition definition );

    void open( imageSource::TextureInfo* info ) override;

    void close() override {}

    bool isOpen() const override { return true; }

    const imageSource::TextureInfo& getInfo() const override { return m_info; }

    CUmemorytype getFillType() const override { return CU_MEMORYTYPE_HOST; }

    bool readTile( char* dest, unsigned int mipLevel, const imageSource::Tile& tile, CUstream stream ) override;

    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, CUstream stream )
        override;

    bool readBaseColor( float4& dest ) override;

  private:
    float4 pixel( unsigned int x, unsigned int y, unsigned int width, unsigned int height ) const;

    PbrtCheckerboardDefinition m_definition{};
    imageSource::TextureInfo   m_info{};
};

}  // namespace demandPbrtScene
