// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ShaderUtil/Transform4.h>
#include <OptiXToolkit/ShaderUtil/vec_printers.h>

#include <ostream>

namespace otk {

inline std::ostream& operator<<( std::ostream& str, const Transform4& val )
{
    return str << "[ " << val.m[0] << ", " << val.m[1] << ", " << val.m[2] << ", " << val.m[3] << " ]";
}

}  // namespace otk
