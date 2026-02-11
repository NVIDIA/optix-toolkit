// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <string>

/// Callback function for demand load logging
typedef void ( *DemandLoadLogCallback )( int level, const char* message );

/// Standard callback that logs to stderr.
void standardDemandLoadLogCallback( int level, const char* message );

/// Logger for demand loading
class DemandLoadLogger
{
  public:
    /// Set the log callback and level. Must be called at most once, before any logging occurs.
    static void setLogFunction( DemandLoadLogCallback callback, int level )
    {
        OTK_ASSERT_MSG( !m_initialized, "setLogFunction must be called only once" );
        m_callback = callback;
        m_logLevel = level;
        m_initialized = true;
    }
    static void log( int level, const std::string& message ) 
    { 
        if( m_callback && level <= m_logLevel )
            m_callback( level, message.c_str() );
    }
    static int getLogLevel() { return m_logLevel; }

  private:
    static DemandLoadLogCallback m_callback;
    static int m_logLevel;
    static bool m_initialized;
};

#define DL_LOG(level, message)                           \
    do {                                                 \
        if( DemandLoadLogger::getLogLevel() >= (level) ) \
            DemandLoadLogger::log( (level), (message) ); \
    } while(0)
