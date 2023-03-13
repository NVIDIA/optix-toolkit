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

#include <OptiXToolkit/DemandLoading/Options.h>

#include "RequestProcessor.h"
#include "RequestQueue.h"
#include "Util/TraceFile.h"

#include <cuda.h>

#include <memory>
#include <thread>
#include <vector>

namespace demandLoading {

class PageTableManager;
class TraceFileWriter;

class ThreadPoolRequestProcessor : public RequestProcessor
{
  public:
    /// Construct request processor, which uses the given PageTableManager to
    /// find the RequestHandler associated with a range of pages.
    ThreadPoolRequestProcessor( std::shared_ptr<PageTableManager> pageTableManager, const Options& options );
    ~ThreadPoolRequestProcessor() override = default;

    /// Start processing requests using the specified number of threads.  Options supplies
    /// the number of specified threads (if zero, std::thread::hardware_concurrency is used),
    /// the trace file and the size of the request queue.
    void start( unsigned int maxThreads );

    /// Stop processing requests, terminating threads.
    void stop();

    /// Add a batch of page requests to the request queue.
    void addRequests( CUstream stream, const unsigned int* pageIds, unsigned int numPageIds, Ticket ticket ) override;

    void recordTexture( std::shared_ptr<imageSource::ImageSource> imageSource, const TextureDescriptor& textureDesc );

private:
    std::shared_ptr<PageTableManager> m_pageTableManager;
    std::unique_ptr<RequestQueue>     m_requests;
    std::vector<std::thread>          m_threads;
    std::unique_ptr<TraceFileWriter>  m_traceFile{};

    // Per-thread worker function.
    void worker();
};

}  // namespace demandLoading
