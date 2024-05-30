//
// Copyright (c) 2023 NVIDIA Corporation.  All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software, related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is strictly
// prohibited.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
// AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
// INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
// SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
// LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
// BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES
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
