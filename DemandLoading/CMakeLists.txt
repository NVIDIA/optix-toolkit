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

include(FetchContent)

FetchContent_Declare(
  Imath
  GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/Imath.git
  GIT_TAG v3.1.5
)
FetchContent_MakeAvailable(Imath)

set( OPENEXR_BUILD_TOOLS OFF CACHE BOOL "Enables building of utility programs" )
set( OPENEXR_INSTALL_EXAMPLES OFF CACHE BOOL "Install OpenEXR examples" )

FetchContent_Declare(
  OpenEXR
  GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openexr.git
  GIT_TAG v3.1.5
)
FetchContent_MakeAvailable(OpenEXR)

# Multiple OpenEXR targets have a compile option (/EHsc) that confuses nvcc.
# We replace it with $<$<COMPILE_LANGUAGE:CXX>:/EHsc>.
foreach(_package OpenEXR::OpenEXR OpenEXR::OpenEXRCore)
  get_target_property(_dependencies ${_package} INTERFACE_LINK_LIBRARIES)
  foreach(_lib ${_package} ${_dependencies})
    if(TARGET ${_lib})
      get_target_property(_alias ${_lib} ALIASED_TARGET)
      if(NOT _alias)
        set(_alias ${_lib})
      endif()
      get_target_property(_options ${_alias} INTERFACE_COMPILE_OPTIONS)
      if(_options)
        set(cxx_flag "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>")
        string(FIND ${_options} ${cxx_flag} has_cxx_flag)
        if(${has_cxx_flag} EQUAL -1)
          string(REPLACE "/EHsc" ${cxx_flag} _options "${_options}")
          set_target_properties(${_alias} PROPERTIES INTERFACE_COMPILE_OPTIONS "${_options}")
        endif()
      endif()
    endif()
  endforeach()
endforeach()

add_library( DemandLoading STATIC
  include/DemandLoading/DemandLoader.h
  include/DemandLoading/DemandTexture.h
  include/DemandLoading/DeviceContext.h
  include/DemandLoading/Options.h
  include/DemandLoading/Resource.h
  include/DemandLoading/Statistics.h
  include/DemandLoading/Texture2D.h
  include/DemandLoading/Texture2DExtended.h
  include/DemandLoading/Texture2DFootprint.h
  include/DemandLoading/TextureDescriptor.h
  include/DemandLoading/TextureSampler.h
  include/DemandLoading/Ticket.h
  include/DemandLoading/TileIndexing.h
  include/ImageSource/CheckerBoardImage.h
  include/ImageSource/CoreEXRReader.h
  include/ImageSource/DeviceConstantImage.h
  include/ImageSource/DeviceConstantImageParams.h
  include/ImageSource/DeviceMandelbrotImage.h
  include/ImageSource/DeviceMandelbrotParams.h
  include/ImageSource/EXRReader.h
  include/ImageSource/ImageSource.h
  include/ImageSource/MultiCheckerImage.h
  include/ImageSource/TextureInfo.h
  src/DemandLoaderImpl.cpp
  src/DemandLoaderImpl.h
  src/DeviceContextImpl.cpp
  src/DeviceContextImpl.h
  src/ImageSource/CheckerBoardImage.cpp
  src/ImageSource/CoreEXRReader.cpp
  src/ImageSource/DeviceConstantImage.cpp
  src/ImageSource/DeviceConstantImageKernels.cu
  src/ImageSource/DeviceMandelbrotImage.cpp
  src/ImageSource/DeviceMandelbrotImageKernels.cu
  src/ImageSource/EXRReader.cpp
  src/ImageSource/Exception.h
  src/ImageSource/ImageSource.cpp
  src/ImageSource/Stopwatch.h
  src/ImageSource/TextureInfo.cpp
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

source_group( "Header Files\\Implementation" FILES
  src/DemandLoaderImpl.h
  src/DeviceContextImpl.h
  src/ImageSource/Exception.h
  src/ImageSource/Stopwatch.h
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

target_compile_definitions( DemandLoading PUBLIC 
  OPTIX_SAMPLE_USE_OPEN_EXR
  OPTIX_SAMPLE_USE_CORE_EXR
  )

set_property(TARGET DemandLoading PROPERTY CUDA_ARCHITECTURES "60")

target_include_directories( DemandLoading
  PUBLIC
  include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${OptiX_INCLUDE}
  PRIVATE
  src
  )

target_link_libraries( DemandLoading PUBLIC
  cuda
  OpenEXR::OpenEXR
  OpenEXR::OpenEXRCore
  )

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

add_subdirectory( tests )
