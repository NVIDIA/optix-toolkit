# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
include(Policies)
include(SetCxxStandard)

project(Memory LANGUAGES C CXX CUDA)

set_cxx_standard(11)
set(CMAKE_CXX_EXTENSIONS OFF)

include(BuildConfig)
include(CTest)
include(FetchOptiX)

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 11.1 REQUIRED )
endif()

if( CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.2 )
  option( OTK_USE_CUDA_MEMORY_POOLS "Use CUDA memory pools and stream-ordered allocators" ON )
else()
  option( OTK_USE_CUDA_MEMORY_POOLS "Use CUDA memory pools and stream-ordered allocators" OFF )
endif()

# The Error library provides uniform error messaging and exception support
# across the CUDA driver API, the CUDA runtime API and the OptiX API.
#
# Including the appropriate API error check header presumes you already have
# a dependency on the relevant API library and therefore we don't add one here.
# If you include the API error checking header and haven't added such a dependency
# yourself you are likely to get a compilation error about not finding includes
# or a link error about not finding dependent symbols.
#
# The ErrorCheck.h header is standalone and depends only on the C++ standard
# library.  You can plug your own error result types into this mechanism by
# specializing a couple template functions.  See ErrorCheck.h for details
# and the provided API headers for examples.
#
otk_add_library(Error INTERFACE)
target_sources(Error
  PUBLIC
  FILE_SET HEADERS
  BASE_DIRS include
  FILES
  include/OptiXToolkit/Error/cuErrorCheck.h
  include/OptiXToolkit/Error/cudaErrorCheck.h
  include/OptiXToolkit/Error/ErrorCheck.h
  include/OptiXToolkit/Error/optixErrorCheck.h
)
target_link_libraries( Error INTERFACE
  CUDA::cuda_driver
  CUDA::cudart
  )
set_property(TARGET Error PROPERTY FOLDER Memory)
add_library(OptiXToolkit::Error ALIAS Error)

install(TARGETS Error
  EXPORT ErrorTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(EXPORT ErrorTargets
  FILE ErrorTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::
  )

# Basic memory allocators and operations.
otk_add_library(Memory INTERFACE)
set_property(TARGET Memory PROPERTY FOLDER Memory)
target_sources(Memory
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/Memory/Allocators.h
  include/OptiXToolkit/Memory/BinnedSuballocator.h
  include/OptiXToolkit/Memory/BitCast.h
  include/OptiXToolkit/Memory/DeviceBuffer.h
  include/OptiXToolkit/Memory/DeviceFixedPool.h
  include/OptiXToolkit/Memory/DeviceRingBuffer.h
  include/OptiXToolkit/Memory/FixedSuballocator.h
  include/OptiXToolkit/Memory/HeapSuballocator.h
  include/OptiXToolkit/Memory/MemoryBlockDesc.h
  include/OptiXToolkit/Memory/MemoryPool.h
  include/OptiXToolkit/Memory/RingSuballocator.h
  include/OptiXToolkit/Memory/SyncVector.h
)
target_include_directories( Memory INTERFACE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
target_link_libraries( Memory INTERFACE
  Error
  CUDA::cuda_driver
  )
if( OTK_USE_CUDA_MEMORY_POOLS )
  target_compile_options( Memory INTERFACE "-DOTK_USE_CUDA_MEMORY_POOLS" )
endif()
add_library(OptiXToolkit::Memory ALIAS Memory)

install(TARGETS Memory
  EXPORT MemoryTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(EXPORT MemoryTargets
  FILE MemoryTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::
  )

# OptiX related memory operations.
otk_add_library(OptiXMemory INTERFACE)
target_sources(OptiXMemory
  PUBLIC
  FILE_SET HEADERS
  BASE_DIRS include
  FILES
  include/OptiXToolkit/OptiXMemory/Builders.h
  include/OptiXToolkit/OptiXMemory/CompileOptions.h
  include/OptiXToolkit/OptiXMemory/Record.h
  include/OptiXToolkit/OptiXMemory/SyncRecord.h
)
target_include_directories(OptiXMemory INTERFACE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
target_link_libraries(OptiXMemory INTERFACE
  OptiX::OptiX
  Memory
  CUDA::cuda_driver
  )
set_target_properties(OptiXMemory PROPERTIES FOLDER Memory)
add_library(OptiXToolkit::OptiXMemory ALIAS OptiXMemory)

install(TARGETS OptiXMemory
  EXPORT OptiXMemoryTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(EXPORT OptiXMemoryTargets
  FILE OptiXMemoryTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::
  )

if( BUILD_TESTING )
  add_subdirectory( tests )
endif()
