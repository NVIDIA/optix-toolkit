// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "DemandPbrtScene/Config.h"

#ifdef OTK_USE_MDL

#include <string>

namespace mi {
namespace neuraylib {

class ICompiled_material;
class IMdl_execution_context;
class INeuray;
class ITransaction;

}  // namespace neuraylib
}  // namespace mi

namespace demandPbrtScene {

struct MdlBsdfCallablePtx
{
    std::string initFunctionName;
    std::string sampleFunctionName;
    std::string evaluateFunctionName;
    std::string pdfFunctionName;
    std::string ptx;
};

MdlBsdfCallablePtx compileMdlBsdfCallablesToPtx( mi::neuraylib::INeuray*                  neuray,
                                                 mi::neuraylib::ITransaction*             transaction,
                                                 const mi::neuraylib::ICompiled_material* compiledMaterial,
                                                 mi::neuraylib::IMdl_execution_context*   context,
                                                 const std::string&                       expressionPath,
                                                 const std::string&                       baseFunctionName );

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL
