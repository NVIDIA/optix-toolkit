// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/PbrtSceneLoader/GoogleLogger.h>

#include <glog/logging.h>

namespace otk {
namespace pbrt {

void GoogleLogger::start( const char* programName )
{
    google::InitGoogleLogging( programName );
    FLAGS_logtostderr = true;
    FLAGS_minloglevel = m_minLogLevel;
}

void GoogleLogger::stop()
{
    google::ShutdownGoogleLogging();
}

void GoogleLogger::info( std::string text, const char *file, int line ) const
{
    if( text.empty() )
        return;
    if( text.back() != '\n' )
        text += '\n';
    google::LogMessage( file, line, google::GLOG_INFO ).stream() << text;
}

void GoogleLogger::warning( std::string text, const char *file, int line ) const
{
    if( text.empty() )
        return;
    if( text.back() != '\n' )
        text += '\n';
    google::LogMessage( file, line, google::GLOG_WARNING ).stream() << text;
}

void GoogleLogger::error( std::string text, const char *file, int line ) const
{
    if( text.empty() )
        return;
    if( text.back() != '\n' )
        text += '\n';
    google::LogMessage( file, line, google::GLOG_ERROR ).stream() << text;
}

}  // namespace pbrt
}  // namespace otk
