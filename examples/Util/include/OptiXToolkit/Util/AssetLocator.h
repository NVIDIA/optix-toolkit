// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <string>

namespace otk {

/// Return the location of an asset in relativeSubDIr with relativePath.
/// Search the locations specified by the environment variable OTK_ASSET_DIR
/// followed by the build-time configuration value of OTK_BINARY_DATA_DIR.
///
/// std::runtime_error is thrown if the asset could not be located.
///
std::string locateAsset( const char* relativeSubDir, const char* relativePath );

inline std::string locateAsset( const std::string& relativeSubDir, const std::string& relativePath )
{
    return locateAsset( relativeSubDir.c_str(), relativePath.c_str() );
}

}  // namespace otk
