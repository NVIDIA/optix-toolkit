// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file RequestError.h
/// Error recorded while handling a demand-load request.

#include <string>

namespace imageSource {
class ImageSource;
}

namespace demandLoading {

/// Which part of demand loading failed.
enum class RequestErrorStage
{
    ImageLoad,      ///< ImageSource::open, base color, or a dense-texture image read
    SamplerCreate,  ///< DemandTextureImpl::init and publishing the sampler
    TextureTile,    ///< readTile, readMipTail, fillTile, or the mip tail
    Other           ///< Resource callback, cascade bookkeeping, or an unexpected failure
};

/// A failure recorded while handling a demand-load request.
///
/// image is the ImageSource passed to createTexture or createUdimTexture, or null when no image
/// is involved. It is non-owning and remains valid while the application or the DemandLoader still
/// holds that image. textureId is ~0u when no texture is involved.
struct RequestError
{
    RequestErrorStage         stage     = RequestErrorStage::Other;
    unsigned int              textureId = ~0u;
    imageSource::ImageSource* image     = nullptr;
    std::string               message;
};

}  // namespace demandLoading
