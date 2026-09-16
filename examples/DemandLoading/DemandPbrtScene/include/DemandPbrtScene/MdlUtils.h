// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Config.h"

#ifdef OTK_USE_MDL

#include <cstddef>
#include <string>
#include <vector>

namespace mi {
namespace neuraylib {

class ICompiled_material;
class IMdl_execution_context;
class ITarget_code;

}  // namespace neuraylib
}  // namespace mi

namespace demandPbrtScene {

struct MdlTargetArgumentBlockParameter
{
    std::string  name;
    unsigned int kind{};
    std::size_t  offset{};
    std::size_t  size{};
};

struct MdlTargetArgumentBlock
{
    std::vector<char>                            data;
    std::vector<MdlTargetArgumentBlockParameter> parameters;
};

std::string describeMdlContextMessages( const mi::neuraylib::IMdl_execution_context* context );

[[noreturn]] void failMdl( const std::string& message,
                           const mi::neuraylib::IMdl_execution_context* context = nullptr );

void requireMdl( bool condition, const std::string& message,
                 const mi::neuraylib::IMdl_execution_context* context = nullptr );

MdlTargetArgumentBlock captureMdlTargetArgumentBlock( const mi::neuraylib::ITarget_code*       targetCode,
                                                      const mi::neuraylib::ICompiled_material* compiledMaterial,
                                                      mi::neuraylib::IMdl_execution_context*   context );

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL
