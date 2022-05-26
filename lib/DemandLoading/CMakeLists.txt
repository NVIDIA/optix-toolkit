#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

add_library( DemandLoading STATIC
  ${CMAKE_SOURCE_DIR}/include/DemandLoading/DemandLoader.h
  ${CMAKE_SOURCE_DIR}/include/DemandLoading/DemandTexture.h
  ${CMAKE_SOURCE_DIR}/include/DemandLoading/DeviceContext.h
  ${CMAKE_SOURCE_DIR}/include/DemandLoading/Options.h
  ${CMAKE_SOURCE_DIR}/include/DemandLoading/Resource.h
  ${CMAKE_SOURCE_DIR}/include/DemandLoading/Statistics.h
  ${CMAKE_SOURCE_DIR}/include/DemandLoading/Texture2D.h
  ${CMAKE_SOURCE_DIR}/include/DemandLoading/Texture2DExtended.h
  ${CMAKE_SOURCE_DIR}/include/DemandLoading/Texture2DFootprint.h
  ${CMAKE_SOURCE_DIR}/include/DemandLoading/TextureDescriptor.h
  ${CMAKE_SOURCE_DIR}/include/DemandLoading/TextureSampler.h
  ${CMAKE_SOURCE_DIR}/include/DemandLoading/Ticket.h
  ${CMAKE_SOURCE_DIR}/include/DemandLoading/TileIndexing.h
  DemandLoaderImpl.cpp
  DemandLoaderImpl.h
  DeviceContextImpl.cpp
  DeviceContextImpl.h
  PageMappingsContext.h
  PageTableManager.h
  PagingSystem.cpp
  PagingSystem.h
  PagingSystemKernels.cu
  PagingSystemKernels.h
  RequestContext.h
  RequestHandler.h
  RequestProcessor.cpp
  RequestProcessor.h
  RequestQueue.cpp
  RequestQueue.h
  ResourceRequestHandler.cpp
  ResourceRequestHandler.h
  Ticket.cpp
  TicketImpl.h
  Memory/Allocators.h
  Memory/AsyncItemPool.h
  Memory/Buffers.h
  Memory/BulkMemory.h
  Memory/BulkPinnedItemPool.h
  Memory/DeviceContextPool.cpp
  Memory/DeviceContextPool.h
  Memory/DeviceMemoryManager.h
  Memory/EventPool.h
  Memory/FixedPool.h
  Memory/ItemPool.h
  Memory/PinnedItemPool.h
  Memory/PinnedMemoryManager.h
  Memory/PinnedRequestContextPool.h
  Memory/SamplerPool.h
  Memory/TileArena.h
  Memory/TilePool.cpp
  Memory/TilePool.h
  Textures/BaseColorRequestHandler.cpp
  Textures/BaseColorRequestHandler.h
  Textures/DemandTextureImpl.cpp
  Textures/DemandTextureImpl.h
  Textures/DenseTexture.cpp
  Textures/DenseTexture.h
  Textures/SamplerRequestHandler.cpp
  Textures/SamplerRequestHandler.h
  Textures/SparseTexture.cpp
  Textures/SparseTexture.h
  Textures/TextureRequestHandler.cpp
  Textures/TextureRequestHandler.h
  Util/CudaCallback.h
  Util/DeviceSet.cpp
  Util/DeviceSet.h
  Util/Exception.h
  Util/Math.h
  Util/MutexArray.h
  Util/NVTXProfiling.h
  Util/Stopwatch.h
  Util/TraceFile.cpp
  Util/TraceFile.h
  )

source_group( "Header Files\\Implementation" FILES
  DemandLoaderImpl.h
  DeviceContextImpl.h
  PageMappingsContext.h
  PageTableManager.h
  PagingSystem.h
  PagingSystemKernels.h
  RequestContext.h
  RequestHandler.h
  RequestProcessor.h
  RequestQueue.h
  ResourceRequestHandler.h
  TicketImpl.h
  Memory/Allocators.h
  Memory/AsyncItemPool.h
  Memory/Buffers.h
  Memory/BulkMemory.h
  Memory/BulkPinnedItemPool.h
  Memory/DeviceContextPool.h
  Memory/DeviceMemoryManager.h
  Memory/EventPool.h
  Memory/FixedPool.h
  Memory/ItemPool.h
  Memory/PinnedItemPool.h
  Memory/PinnedMemoryManager.h
  Memory/PinnedRequestContextPool.h
  Memory/SamplerPool.h
  Memory/TileArena.h
  Memory/TilePool.h
  Textures/BaseColorRequestHandler.h
  Textures/DemandTextureImpl.h
  Textures/DenseTexture.h
  Textures/SamplerRequestHandler.h
  Textures/SparseTexture.h
  Textures/TextureRequestHandler.h
  Util/CudaCallback.h
  Util/DeviceSet.h
  Util/Exception.h
  Util/Math.h
  Util/MutexArray.h
  Util/NVTXProfiling.h
  Util/Stopwatch.h
  Util/TraceFile.h
  )

target_include_directories( DemandLoading PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_link_libraries( DemandLoading PUBLIC
  ImageSource
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )

set_property(TARGET DemandLoading PROPERTY CUDA_ARCHITECTURES "60")
set_property(TARGET DemandLoading PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# NVTX Profiling
option( DEMAND_LOADING_USE_NVTX "Enable NVTX profiling" OFF )
if( DEMAND_LOADING_USE_NVTX )
    find_package( NVTX )
    if( NVTX_FOUND )
        message( "-- Found NVTX: Enabling NVTX profiling in the DemandLoading library" )
        include_directories( ${NVTX_INCLUDE} )
        target_compile_definitions( DemandLoading PUBLIC ENABLE_NVTX_PROFILING )
    else()
        message( "-- Not Found NVTX: Disabling NVTX profiling in the DemandLoading library" )
    endif()
endif()
