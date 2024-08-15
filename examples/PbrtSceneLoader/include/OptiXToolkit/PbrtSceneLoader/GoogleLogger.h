// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/PbrtSceneLoader/Logger.h>

namespace otk {
namespace pbrt {

class GoogleLogger : public Logger
{
  public:
    /// Messages at or above the given level are logged.  (0=info, 1=warning, 2=error, 3=fatal)
    GoogleLogger( int minLogLevel )
        : m_minLogLevel( minLogLevel )
    {
    }
    
    ~GoogleLogger() override = default;

    void start( const char* programName ) override;
    void stop() override;
    void info( std::string text, const char *file, int line ) const override;
    void warning( std::string text, const char *file, int line ) const override;
    void error( std::string text, const char* file, int line ) const override;

  private:
    const int m_minLogLevel{};
};

}  // namespace pbrt
}  // namespace otk
