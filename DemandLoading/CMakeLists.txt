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

# Common CMake files are included from a top-level directory, but otherwise
# this directory can be built as a standalone project.
set(OTK_SOURCE_DIR ".." CACHE PATH "OptiX Toolkit source directory")

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
include(${OTK_SOURCE_DIR}/CMake/Policies.cmake)

project(DemandLoading LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(${OTK_SOURCE_DIR}/CMake/BuildConfig.cmake)

find_package( CUDAToolkit 11.1 REQUIRED )

if ( CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.2 )
  option( OTK_USE_CUDA_MEMORY_POOLS "Use CUDA memory pools and stream-ordered allocators" ON )
else()
  option( OTK_USE_CUDA_MEMORY_POOLS "Use CUDA memory pools and stream-ordered allocators" ON )
endif()

# Set OptiX_INSTALL_DIR to the root of the OptiX SDK when configuring CMake.
set(OptiX_INSTALL_DIR "OptiX_INSTALL_DIR-NOTFOUND" CACHE PATH "Path to OptiX installed location.")
list(APPEND CMAKE_MODULE_PATH ${OTK_SOURCE_DIR}/CMake)
find_package(OptiX REQUIRED)

if(PROJECT_IS_TOP_LEVEL)
  find_package(OptiXToolkit REQUIRED)
endif()

option( BUILD_SHARED_LIBS "Build using shared libraries" ON )

add_library( DemandLoading
  src/DemandLoaderImpl.cpp
  src/DemandLoaderImpl.h
  src/DeviceContextImpl.cpp
  src/DeviceContextImpl.h
  src/Memory/Allocators.h
  src/Memory/AsyncItemPool.h
  src/Memory/Buffers.h
  src/Memory/BulkMemory.h
  src/Memory/BulkPinnedItemPool.h
  src/Memory/DeviceContextPool.cpp
  src/Memory/DeviceContextPool.h
  src/Memory/DeviceMemoryManager.h
  src/Memory/EventPool.h
  src/Memory/FixedPool.h
  src/Memory/ItemPool.h
  src/Memory/PinnedItemPool.h
  src/Memory/PinnedMemoryManager.h
  src/Memory/PinnedRequestContextPool.h
  src/Memory/SamplerPool.h
  src/Memory/TileArena.h
  src/Memory/TilePool.cpp
  src/Memory/TilePool.h
  src/PageMappingsContext.h
  src/PageTableManager.h
  src/PagingSystem.cpp
  src/PagingSystem.h
  src/PagingSystemKernels.cu
  src/PagingSystemKernels.h
  src/RequestContext.h
  src/RequestHandler.h
  src/RequestProcessor.cpp
  src/RequestProcessor.h
  src/RequestQueue.cpp
  src/RequestQueue.h
  src/ResourceRequestHandler.cpp
  src/ResourceRequestHandler.h
  src/Textures/BaseColorRequestHandler.cpp
  src/Textures/BaseColorRequestHandler.h
  src/Textures/DemandTextureImpl.cpp
  src/Textures/DemandTextureImpl.h
  src/Textures/DenseTexture.cpp
  src/Textures/DenseTexture.h
  src/Textures/SamplerRequestHandler.cpp
  src/Textures/SamplerRequestHandler.h
  src/Textures/SparseTexture.cpp
  src/Textures/SparseTexture.h
  src/Textures/TextureRequestHandler.cpp
  src/Textures/TextureRequestHandler.h
  src/Ticket.cpp
  src/TicketImpl.h
  src/Util/CudaCallback.h
  src/Util/DeviceSet.cpp
  src/Util/DeviceSet.h
  src/Util/Exception.h
  src/Util/Math.h
  src/Util/MutexArray.h
  src/Util/NVTXProfiling.h
  src/Util/Stopwatch.h
  src/Util/TraceFile.cpp
  src/Util/TraceFile.h
  )

if(NOT PROJECT_IS_TOP_LEVEL)
  add_library(OptiXToolkit::DemandLoading ALIAS DemandLoading)
endif()

target_sources(DemandLoading
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/DemandLoading/DemandLoader.h
  include/OptiXToolkit/DemandLoading/DemandTexture.h
  include/OptiXToolkit/DemandLoading/DeviceContext.h
  include/OptiXToolkit/DemandLoading/Options.h
  include/OptiXToolkit/DemandLoading/Paging.h
  include/OptiXToolkit/DemandLoading/Resource.h
  include/OptiXToolkit/DemandLoading/Statistics.h
  include/OptiXToolkit/DemandLoading/Texture2DExtended.h
  include/OptiXToolkit/DemandLoading/Texture2DFootprint.h
  include/OptiXToolkit/DemandLoading/Texture2D.h
  include/OptiXToolkit/DemandLoading/TextureDescriptor.h
  include/OptiXToolkit/DemandLoading/TextureSampler.h
  include/OptiXToolkit/DemandLoading/Ticket.h
  include/OptiXToolkit/DemandLoading/TileIndexing.h
)

source_group( "Header Files\\Implementation" FILES
  src/DemandLoaderImpl.h
  src/DeviceContextImpl.h
  src/Memory/Allocators.h
  src/Memory/AsyncItemPool.h
  src/Memory/Buffers.h
  src/Memory/BulkMemory.h
  src/Memory/BulkPinnedItemPool.h
  src/Memory/DeviceContextPool.h
  src/Memory/DeviceMemoryManager.h
  src/Memory/EventPool.h
  src/Memory/FixedPool.h
  src/Memory/ItemPool.h
  src/Memory/PinnedItemPool.h
  src/Memory/PinnedMemoryManager.h
  src/Memory/PinnedRequestContextPool.h
  src/Memory/SamplerPool.h
  src/Memory/TileArena.h
  src/Memory/TilePool.h
  src/PageMappingsContext.h
  src/PageTableManager.h
  src/PagingSystem.h
  src/PagingSystemKernels.h
  src/RequestContext.h
  src/RequestHandler.h
  src/RequestProcessor.h
  src/RequestQueue.h
  src/ResourceRequestHandler.h
  src/Textures/BaseColorRequestHandler.h
  src/Textures/DemandTextureImpl.h
  src/Textures/DenseTexture.h
  src/Textures/SamplerRequestHandler.h
  src/Textures/SparseTexture.h
  src/Textures/TextureRequestHandler.h
  src/TicketImpl.h
  src/Util/CudaCallback.h
  src/Util/DeviceSet.h
  src/Util/Exception.h
  src/Util/Math.h
  src/Util/MutexArray.h
  src/Util/NVTXProfiling.h
  src/Util/Stopwatch.h
  src/Util/TraceFile.h
  )

target_include_directories( DemandLoading
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${OptiX_INCLUDE_DIR}
  PRIVATE
  src
  )

target_link_libraries( DemandLoading
  PUBLIC
  OptiXToolkit::ImageSource
  OptiX::OptiX
  CUDA::cuda_driver
  ${CMAKE_DL_LIBS}  
  )

set_target_properties(DemandLoading PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

if( OTK_USE_CUDA_MEMORY_POOLS )
  target_compile_options( DemandLoading PRIVATE "-DOTK_USE_CUDA_MEMORY_POOLS" )
endif()

# NVTX Profiling
option( DEMAND_LOADING_USE_NVTX "Enable NVTX profiling" OFF )
if( DEMAND_LOADING_USE_NVTX )
  target_compile_definitions( DemandLoading PUBLIC ENABLE_NVTX_PROFILING )
endif()

install(TARGETS DemandLoading
  EXPORT DemandLoadingTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(EXPORT DemandLoadingTargets
  FILE DemandLoadingTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::
  )

set( OTK_BUILD_TESTS ON CACHE BOOL "Enable build of OptiXToolkit tests" )
if( OTK_BUILD_TESTS )
  add_subdirectory( tests )
endif()

if( PROJECT_IS_TOP_LEVEL )
  set( OTK_BUILD_DOCS ON CACHE BOOL "Enable build of OptiXToolkit documentation" )
  if( OTK_BUILD_DOCS )
    add_subdirectory( docs/API )
  endif()
endif()
