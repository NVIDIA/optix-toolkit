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

#include <OptiXToolkit/DemandLoading/Options.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/ImageSource/EXRReader.h>

#include <cassert>

namespace demandLoading {

enum RecordType
{
    OPTIONS,
    TEXTURE,
    REQUESTS
};

// Check that the current CUDA context matches the one associated with the given stream
// and return the associated device index.
static unsigned int getDeviceIndex( CUstream /*stream*/ )
{
    // Get the current CUDA context.
    CUcontext cudaContext, streamContext;
    OTK_ERROR_CHECK( cuCtxGetCurrent( &cudaContext ) );
    OTK_ERROR_CHECK( cuCtxGetCurrent( &streamContext ) );
    OTK_ASSERT_MSG( cudaContext == streamContext,
                       "The current CUDA context must match the one associated with the given stream" );

    // Get the device index from the CUDA context.
    CUdevice device;
    OTK_ERROR_CHECK( cuCtxGetDevice( &device ) );
    return static_cast<unsigned int>( device );
}

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
    write( getDeviceIndex( CUstream{0} ) );
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
}

void TraceFileWriter::recordTexture( std::shared_ptr<imageSource::ImageSource> imageSource, const TextureDescriptor& desc )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    // For now, only EXRReader can be serialized.
    std::shared_ptr<imageSource::EXRReader> exrReader( std::dynamic_pointer_cast<imageSource::EXRReader>( imageSource ) );
    if( !exrReader )
        throw std::runtime_error( "Cannot serialize ImageSource (expected EXRReader)" );

    write( TEXTURE );
    write( getDeviceIndex( CUstream{0} ) );
    exrReader->serialize( m_file );

    // Serialize TextureDescriptor.
    write( desc.addressMode[0] );
    write( desc.addressMode[1] );
    write( desc.filterMode );
    write( desc.mipmapFilterMode );
    write( desc.maxAnisotropy );
    write( desc.flags );
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

void TraceFileWriter::recordRequests( CUstream stream, const unsigned int* pageIds, unsigned int numPageIds )
{
    if( !m_file )
        return;

    std::unique_lock<std::mutex> lock( m_mutex );
    unsigned int deviceIndex = getDeviceIndex( stream );
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
        unsigned int deviceIndex = 0;
        read( &deviceIndex );
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
        return options;
    }

    void replay( std::vector<DemandLoader*>& loaders )
    {
        while( true )
        {
            RecordType recordType;
            read( &recordType );
            if( m_file.eof() )
                break;
            if( recordType == TEXTURE )
            {
                replayCreateTexture( loaders );
            }
            else if( recordType == REQUESTS )
            {
                replayRequests( loaders );
            }
            else
            {
                throw std::runtime_error( "Unknown record type in trace file" );
            }
        }
    }

    CUcontext getContext( unsigned int deviceIndex )
    {
        if( deviceIndex >= m_contexts.size() )
        {
            m_contexts.resize( deviceIndex + 1 );
        }
        if( !m_contexts[deviceIndex] )
        {
            CUdevice device;
            OTK_ERROR_CHECK( cuDeviceGet( &device, deviceIndex ) );
            OTK_ERROR_CHECK( cuCtxCreate( &m_contexts[deviceIndex], 0, device ) );
        }
        return m_contexts[deviceIndex];
    }

  private:
    std::ifstream          m_file;
    std::vector<CUcontext> m_contexts;
    std::vector<CUstream>  m_streams;

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
            throw std::runtime_error( stream.str().c_str() );
        }
        read( option );
    }

    void replayCreateTexture( std::vector<DemandLoader*>& loaders )
    {
        unsigned int deviceIndex = 0;
        read( &deviceIndex );

        // FIXME: The image sources can be shared between devices and variant textures.
        // This should be handled.
        std::shared_ptr<imageSource::ImageSource> imageSource( imageSource::EXRReader::deserialize( m_file ) );

        TextureDescriptor desc;
        read( &desc.addressMode[0] );
        read( &desc.addressMode[1] );
        read( &desc.filterMode );
        read( &desc.mipmapFilterMode );
        read( &desc.maxAnisotropy );
        read( &desc.flags );

        OTK_ERROR_CHECK( cuCtxSetCurrent( m_contexts[deviceIndex] ) );
        loaders[deviceIndex]->createTexture( imageSource, desc );
    }

    CUstream getStream( unsigned int deviceIndex, unsigned int streamId )
    {
        if( streamId < m_streams.size() )
            return m_streams[streamId];

        if( streamId != m_streams.size() )
            throw std::runtime_error( "Unexpected stream id in page request trace file" );

        CUcontext context = getContext( deviceIndex );
        OTK_ERROR_CHECK( cuCtxSetCurrent( context ) );

        CUstream stream;
        OTK_ERROR_CHECK( cuStreamCreate( &stream, 0U ) );
        m_streams.push_back( stream );
        return stream;
    }

    void replayRequests( std::vector<DemandLoader*>& loaders )
    {
        unsigned int deviceIndex;
        read( &deviceIndex );

        unsigned int streamId;
        read( &streamId );

        unsigned int numPageIds;
        read( &numPageIds );

        std::vector<unsigned int> pageIds( numPageIds );
        read( pageIds.data(), numPageIds );

        CUcontext context = getContext( deviceIndex );
        OTK_ERROR_CHECK( cuCtxSetCurrent( context ) );
        CUstream stream = getStream( deviceIndex, streamId );

        // Downcast demand loader, since trace file playback relies on internal interface.
        DemandLoaderImpl* loaderImpl = dynamic_cast<DemandLoaderImpl*>( loaders[deviceIndex] );
        assert( loaderImpl );

        Ticket ticket = loaderImpl->replayRequests( stream, pageIds.data(), numPageIds );
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

    // Create demand loaders for each device.
    int numDevices;
    OTK_ERROR_CHECK( cuDeviceGetCount( &numDevices ) );
    std::vector<DemandLoader*> loaders;
    for( int deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex )
    {
        CUcontext context = reader.getContext( deviceIndex );
        OTK_ERROR_CHECK( cuCtxSetCurrent( context ) );
        loaders.push_back( createDemandLoader( options ) );
    }

    // Replay the trace file.
    reader.replay( loaders );

    // FIXME: Get stats from all loaders
    Statistics stats = loaders[0]->getStatistics();

    // Clean up.
    for( int deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex )
    {
        destroyDemandLoader( loaders[deviceIndex] );
    }

    return stats;
}


}  // namespace demandLoading
