# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
include(Policies)
include(SetCxxStandard)

project(DemandLoading LANGUAGES C CXX CUDA)
set_cxx_standard(11)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(BuildConfig)
include(embed_cuda)
include(FetchOpenEXR)
include(FetchOptiX)
include(CTest)

find_package( CUDAToolkit 11.1 REQUIRED )

embed_cuda(CONST RELOCATABLE
    PTX
    HEADER DemandLoadingKernelsCuda.h
    OUTPUT_TARGET DemandLoadingKernels
    FOLDER DemandLoading
    INCLUDES include
    SOURCES src/PagingSystemKernels.cu
)

otk_add_library( DemandLoading
  src/CascadeRequestFilter.cpp
  src/CascadeRequestFilter.h
  src/DemandLoaderImpl.cpp
  src/DemandLoaderImpl.h
  src/DemandPageLoaderImpl.cpp
  src/DemandPageLoaderImpl.h
  src/DeviceContextImpl.cpp
  src/DeviceContextImpl.h
  src/Memory/DeviceMemoryManager.cpp
  src/Memory/DeviceMemoryManager.h
  src/PageMappingsContext.h
  src/PageTableManager.h
  src/PagingSystem.cpp
  src/PagingSystem.h
  src/PagingSystemKernels.cpp
  src/PagingSystemKernels.h
  src/RequestContext.h
  src/RequestHandler.h
  src/RequestQueue.cpp
  src/RequestQueue.h
  src/ResourceRequestHandler.cpp
  src/ResourceRequestHandler.h
  src/Textures/CascadeRequestHandler.cpp
  src/Textures/CascadeRequestHandler.h
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
  src/ThreadPoolRequestProcessor.cpp
  src/ThreadPoolRequestProcessor.h
  src/Ticket.cpp
  src/TicketImpl.h
  src/TransferBufferDesc.h
  src/Util/ContextSaver.h
  src/Util/CudaCallback.h
  src/Util/CudaContext.h
  src/Util/Math.h
  src/Util/MutexArray.h
  src/Util/NVTXProfiling.h
  src/Util/Stopwatch.h
  )
set_property(TARGET DemandLoading PROPERTY FOLDER DemandLoading)

add_library(OptiXToolkit::DemandLoading ALIAS DemandLoading)

target_sources(DemandLoading
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/DemandLoading/DemandLoader.h
  include/OptiXToolkit/DemandLoading/DemandPageLoader.h
  include/OptiXToolkit/DemandLoading/DemandTexture.h
  include/OptiXToolkit/DemandLoading/DeviceContext.h
  include/OptiXToolkit/DemandLoading/LRU.h
  include/OptiXToolkit/DemandLoading/Options.h
  include/OptiXToolkit/DemandLoading/Paging.h
  include/OptiXToolkit/DemandLoading/RequestFilter.h
  include/OptiXToolkit/DemandLoading/RequestProcessor.h
  include/OptiXToolkit/DemandLoading/Resource.h
  include/OptiXToolkit/DemandLoading/SparseTextureDevices.h
  include/OptiXToolkit/DemandLoading/Statistics.h
  include/OptiXToolkit/DemandLoading/Texture2D.h
  include/OptiXToolkit/DemandLoading/Texture2DCubic.h
  include/OptiXToolkit/DemandLoading/Texture2DExtended.h
  include/OptiXToolkit/DemandLoading/Texture2DFootprint.h
  include/OptiXToolkit/DemandLoading/TextureCascade.h
  include/OptiXToolkit/DemandLoading/TextureDescriptor.h
  include/OptiXToolkit/DemandLoading/TextureSampler.h
  include/OptiXToolkit/DemandLoading/Ticket.h
  include/OptiXToolkit/DemandLoading/TileIndexing.h
)

source_group( "Header Files\\Implementation" FILES
  src/CascadeRequestFilter.h
  src/DemandLoaderImpl.h
  src/DemandPageLoaderImpl.h
  src/DeviceContextImpl.h
  src/Memory/DeviceMemoryManager.h
  src/PageMappingsContext.h
  src/PageTableManager.h
  src/PagingSystem.h
  src/PagingSystemKernels.h
  src/RequestContext.h
  src/RequestHandler.h
  src/RequestQueue.h
  src/ResourceRequestHandler.h
  src/Textures/CascadeRequestHandler.h
  src/Textures/DemandTextureImpl.h
  src/Textures/DenseTexture.h
  src/Textures/SamplerRequestHandler.h
  src/Textures/SparseTexture.h
  src/Textures/TextureRequestHandler.h
  src/ThreadPoolRequestProcessor.h
  src/TicketImpl.h
  src/TransferBufferDesc.h
  src/Util/ContextSaver.h
  src/Util/CudaCallback.h
  src/Util/CudaContext.h
  src/Util/Math.h
  src/Util/MutexArray.h
  src/Util/NVTXProfiling.h
  src/Util/Stopwatch.h
  )

target_include_directories( DemandLoading
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  PRIVATE
  ${OptiX_INCLUDE_DIR}
  src
  )

target_link_libraries(DemandLoading PRIVATE DemandLoadingKernels)
target_link_libraries( DemandLoading
  PUBLIC
  ImageSource
  ShaderUtil
  OptiXToolkit::Memory
  OptiX::OptiX
  CUDA::cuda_driver
  )

set_target_properties(DemandLoading PROPERTIES
  FOLDER DemandLoading
  WINDOWS_EXPORT_ALL_SYMBOLS ON)

if( OTK_USE_CUDA_MEMORY_POOLS )
  target_compile_options( DemandLoading PRIVATE "-DOTK_USE_CUDA_MEMORY_POOLS" )
endif()

# NVTX Profiling
option( OTK_DEMAND_LOADING_USE_NVTX "Enable NVTX profiling" OFF )
if( OTK_DEMAND_LOADING_USE_NVTX )
  target_compile_definitions( DemandLoading PUBLIC ENABLE_NVTX_PROFILING )
endif()

install(TARGETS DemandLoading DemandLoadingKernels
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

if( BUILD_TESTING )
  add_subdirectory( tests )
endif()

if( PROJECT_IS_TOP_LEVEL )
  set( OTK_BUILD_DOCS ON CACHE BOOL "Enable build of OptiXToolkit documentation" )
  if( OTK_BUILD_DOCS )
    add_subdirectory( docs/API )
  endif()
endif()
