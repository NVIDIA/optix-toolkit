//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <DemandLoading/Statistics.h>

#include <cuda.h>

#include <fstream>
#include <map>
#include <memory>
#include <mutex>

namespace imageSource {
class ImageSource;
}

namespace demandLoading {

class DemandLoader;
struct TextureDescriptor;
struct Options;

class TraceFileWriter
{
  public:
    /// Create trace file, opening the specified file.  Throws an exception on error.
    TraceFileWriter( const char* filename );

    /// Close the trace file.
    ~TraceFileWriter();

    /// Record demand loading options.
    void recordOptions( const Options& options );

    /// Record createTexture call.
    void recordTexture( std::shared_ptr<imageSource::ImageSource> imageSource, const TextureDescriptor& desc );

    /// Record a batch of page requests.
    void recordRequests( unsigned int deviceIndex, CUstream stream, const unsigned int* pageIds, unsigned int numPageIds );

  private:
    mutable std::mutex m_mutex;
    std::ofstream      m_file;
    std::map<CUstream, unsigned int> m_streamIds;
    unsigned int m_nextStreamId = 0;

    template <typename T>
    void write( const T& value )
    {
        m_file.write( reinterpret_cast<const char*>( &value ), sizeof( T ) );
    }

    template <typename T>
    void write( const T* value, size_t count )
    {
        m_file.write( reinterpret_cast<const char*>( value ), count * sizeof( T ) );
    }

    template <typename T>
    void writeOption( const std::string& name, const T& value )
    {
        write( name );
        write( value );
    }

    // CUDA streams are assigned integer identifiers as they are encountered.
    unsigned int getStreamId( CUstream stream );
};

/// Replay the specified trace file.  Throws an exception on error.
Statistics replayTraceFile( const char* filename );

}  // namespace demandLoading
