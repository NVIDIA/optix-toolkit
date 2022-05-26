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

#include "TraceFile.h"
#include "DemandLoaderImpl.h"
#include "Util/Exception.h"

#include <DemandLoading/Options.h>
#include <DemandLoading/TextureDescriptor.h>

#ifdef OPTIX_SAMPLE_USE_OPEN_EXR
#include <ImageSource/EXRReader.h>
#endif

#include <cassert>

namespace demandLoading {

enum RecordType
{
    OPTIONS,
    TEXTURE,
    REQUESTS
};

TraceFileWriter::TraceFileWriter( const char* filename )
    : m_file( filename, std::ios::out | std::ios::binary )
{
}

TraceFileWriter::~TraceFileWriter()
{
    m_file.close();
}

template <>
void TraceFileWriter::write( const std::string& str )
{
    write( str.size() );
    write( str.data(), str.size() );
}

void TraceFileWriter::recordOptions( const Options& options )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    write( OPTIONS );
    writeOption( "numPages", options.numPages );
    writeOption( "numPageTableEntries", options.numPageTableEntries );
    writeOption( "maxRequestedPages", options.maxRequestedPages );
    writeOption( "maxFilledPages", options.maxFilledPages );
    writeOption( "maxStalePages", options.maxStalePages );
    writeOption( "maxEvictablePages", options.maxEvictablePages );
    writeOption( "maxInvalidatedPages", options.maxInvalidatedPages );
    writeOption( "maxStagedPages", options.maxStagedPages );
    writeOption( "useLruTable", options.useLruTable );
    writeOption( "maxTexMemPerDevice", options.maxTexMemPerDevice );
    writeOption( "maxPinnedMemory", options.maxPinnedMemory );
    writeOption( "maxThreads", options.maxThreads );
    writeOption( "maxActiveStreams", options.maxActiveStreams );
}

void TraceFileWriter::recordTexture( std::shared_ptr<imageSource::ImageSource> imageSource, const TextureDescriptor& desc )
{
    std::unique_lock<std::mutex> lock( m_mutex );


#ifdef OPTIX_SAMPLE_USE_OPEN_EXR
    // For now, only EXRReader can be serialized.
    std::shared_ptr<imageSource::EXRReader> exrReader( std::dynamic_pointer_cast<imageSource::EXRReader>( imageSource ) );
    if( !exrReader )
        throw Exception( "Cannot serialize ImageSource (expected EXRReader)" );

    write( TEXTURE );
    exrReader->serialize( m_file );

    // Serialize TextureDescriptor.
    write( desc.addressMode[0] );
    write( desc.addressMode[1] );
    write( desc.filterMode );
    write( desc.mipmapFilterMode );
    write( desc.maxAnisotropy );
    write( desc.flags );
#else
    throw Exception( "Cannot serialize ImageSource (EXRReader not available)" );
#endif
}

// CUDA streams are assigned integer identifiers as they are encountered.
unsigned int TraceFileWriter::getStreamId( CUstream stream )
{
    auto it = m_streamIds.find( stream );
    if( it != m_streamIds.end() )
        return it->second;
    unsigned int streamId = m_nextStreamId++;
    m_streamIds[stream]   = streamId;
    return streamId;
}

void TraceFileWriter::recordRequests( unsigned int deviceIndex, CUstream stream, const unsigned int* pageIds, unsigned int numPageIds )
{
    if( !m_file )
        return;

    std::unique_lock<std::mutex> lock( m_mutex );
    unsigned int streamId = getStreamId( stream ) ;

    write( REQUESTS );
    write( deviceIndex );
    write( streamId );
    write( numPageIds );
    write( pageIds, numPageIds );
}

class TraceFileReader
{
  public:
    TraceFileReader( const char* filename )
        : m_file( filename, std::ios::in | std::ios::binary )
    {
    }

    Options readOptions()
    {
        RecordType recordType;
        read( &recordType );
        assert( recordType == OPTIONS );

        Options options;
        readOption( "numPages", &options.numPages );
        readOption( "numPageTableEntries", &options.numPageTableEntries );
        readOption( "maxRequestedPages", &options.maxRequestedPages );
        readOption( "maxFilledPages", &options.maxFilledPages );
        readOption( "maxStalePages", &options.maxStalePages );
        readOption( "maxEvictablePages", &options.maxEvictablePages );
        readOption( "maxInvalidatedPages", &options.maxInvalidatedPages );
        readOption( "maxStagedPages", &options.maxStagedPages );
        readOption( "useLruTable", &options.useLruTable );
        readOption( "maxTexMemPerDevice", &options.maxTexMemPerDevice );
        readOption( "maxPinnedMemory", &options.maxPinnedMemory );
        readOption( "maxThreads", &options.maxThreads );
        readOption( "maxActiveStreams", &options.maxActiveStreams );
        return options;
    }

    void replay( DemandLoader* loader )
    {
        while( true )
        {
            RecordType recordType;
            read( &recordType );
            if( m_file.eof() )
                break;
            if( recordType == TEXTURE )
            {
                replayCreateTexture( loader );
            }
            else if( recordType == REQUESTS )
            {
                replayRequests( loader );
            }
            else
            {
                throw Exception( "Unknown record type in trace file" );
            }
        }
    }

  private:
    std::ifstream         m_file;
    std::vector<CUstream> m_streams;

    template <typename T>
    void read( T* dest )
    {
        m_file.read( reinterpret_cast<char*>( dest ), sizeof( T ) );
    }

    std::string readString()
    {
        size_t size;
        read( &size );
        std::vector<char> buffer( size );
        m_file.read( buffer.data(), size );
        return std::string( buffer.data(), size );
    }

    template <typename T>
    void read( T* dest, unsigned int count )
    {
        m_file.read( reinterpret_cast<char*>( dest ), count * sizeof( T ) );
    }

    template <typename T>
    void readOption( const std::string& expected, T* option )
    {
        std::string found = readString();
        if( found != expected )
        {
            std::stringstream stream;
            stream << "Error reading option from trace file.  Expected " << expected << ", found " << found;
            throw Exception( stream.str().c_str() );
        }
        read( option );
    }

    void replayCreateTexture( DemandLoader* loader )
    {
#ifdef OPTIX_SAMPLE_USE_OPEN_EXR
        std::shared_ptr<imageSource::ImageSource> imageSource( imageSource::EXRReader::deserialize( m_file ) );

        TextureDescriptor desc;
        read( &desc.addressMode[0] );
        read( &desc.addressMode[1] );
        read( &desc.filterMode );
        read( &desc.mipmapFilterMode );
        read( &desc.maxAnisotropy );
        read( &desc.flags );

        loader->createTexture( imageSource, desc );
#else
        throw Exception("Cannot deserialize ImageSource (EXRReader is not available)");
#endif
    }

    CUstream getStream( unsigned int deviceIndex, unsigned int streamId )
    {
        if( streamId < m_streams.size() )
            return m_streams[streamId];

        if( streamId != m_streams.size() )
            throw Exception( "Unexpected stream id in page request trace file" );

        DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );
        CUstream stream;
        DEMAND_CUDA_CHECK( cuStreamCreate( &stream, 0U ) );
        m_streams.push_back( stream );
        return stream;
    }

    void replayRequests( DemandLoader* loader )
    {
        unsigned int deviceIndex;
        read( &deviceIndex );

        unsigned int streamId;
        read( &streamId );

        unsigned int numPageIds;
        read( &numPageIds );

        std::vector<unsigned int> pageIds( numPageIds );
        read( pageIds.data(), numPageIds );

        // Downcast demand loader, since trace file playback relies on internal interface.
        DemandLoaderImpl* loaderImpl = dynamic_cast<DemandLoaderImpl*>( loader );
        assert( loaderImpl );

        if( !loaderImpl->isActiveDevice( deviceIndex ) )
            throw Exception( "Required device is not present for request trace playback." );

        CUstream stream = getStream( deviceIndex, streamId );
        Ticket ticket = loaderImpl->replayRequests( deviceIndex, stream, pageIds.data(), numPageIds );
        ticket.wait();
    }
};

Statistics replayTraceFile( const char* filename )
{
    // Open the trace file.  Throws an exception if an error occurs.
    TraceFileReader reader( filename );

    // Read options.
    Options options = reader.readOptions();
    options.traceFile = "";

    // Create demand loader.
    DemandLoader* loader = createDemandLoader( options );

    // Replay the trace file.
    reader.replay( loader );
    Statistics stats = loader->getStatistics();

    // Clean up.
    destroyDemandLoader( loader );

    return stats;
}


}  // namespace demandLoading
