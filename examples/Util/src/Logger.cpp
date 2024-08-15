// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/Util/Logger.h>

#include <cctype>
#include <iomanip>
#include <iostream>
#include <string>

namespace otk {
namespace util {

using uint_t = unsigned int;

static void contextLog( uint_t level, const char* tag, const char* text, void* /*cbdata */ )
{
    std::string message{ text };
    while( !message.empty() && std::isspace( message.back() ) )
        message.pop_back();
    if( message.empty() )
        return;
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << '\n';
}

void setLogger( OptixDeviceContextOptions& options )
{
    options.logCallbackFunction = contextLog;
    options.logCallbackLevel    = 4;
}

}  // namespace util
}  // namespace otk
