// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "MeshReader.h"

#include <rply/rply.h>

#include <string>

namespace ply {

/// Used to scan a PLY file and obtain aggregate information
/// (bounding box, number of faces, number of vertices, etc.).
class InfoReader : public ::otk::pbrt::MeshInfoReader
{
  public:
    InfoReader()           = default;
    ~InfoReader() override = default;

    otk::pbrt::MeshInfo read( const std::string& filename ) override;

    otk::pbrt::MeshLoaderPtr getLoader( const std::string& filename ) override;

private:
    static int s_readVertex( p_ply_argument argument );
    int        readVertex( p_ply_argument argument, long index );
    static void s_errorCallback( p_ply ply, const char* message );

    otk::pbrt::MeshInfo m_meshInfo{};
    bool                m_firstVertex[3]{};
};

}  // namespace ply
