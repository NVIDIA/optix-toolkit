# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(BuildConfig)
include(FetchOptiX)
include(FetchRapidJSON)

# Check OptiX version (9.0.0 = 90000)
if(OPTIX_VERSION LESS 90000)
  message(STATUS "Skipping NeuralTextures: requires OptiX 9.0.0 or newer. Found version ${OPTIX_VERSION}.")
  return()
endif()

otk_add_library( NeuralTextures
  src/NeuralTextureSource.cpp
  src/NtcImageReader.cpp
)

set_property(TARGET NeuralTextures PROPERTY FOLDER NeuralTextures)

add_library(OptiXToolkit::NeuralTextures ALIAS NeuralTextures)

target_sources(NeuralTextures
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
  FILES
  include/OptiXToolkit/NeuralTextures/NeuralTextureSource.h
  include/OptiXToolkit/NeuralTextures/NtcImageReader.h
  include/OptiXToolkit/NeuralTextures/InferenceDataOptix.h
  include/OptiXToolkit/NeuralTextures/InferenceConstants.h
  include/OptiXToolkit/NeuralTextures/InferenceOptix.h
)

target_include_directories( NeuralTextures
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${OptiX_INCLUDE_DIR}
  )

target_link_libraries( NeuralTextures
  PUBLIC
  OptiXToolkit::ImageSource
  CUDA::cuda_driver
  PRIVATE
  OptiXToolkit::Error
  )

# Link RapidJSON (header-only library)
# RapidJSON doesn't create a target in FetchContent, so we need to add its include directory manually
# Using PUBLIC because NtcImageReader.h includes rapidjson headers
if(RapidJSON_SOURCE_DIR)
  target_include_directories(NeuralTextures PUBLIC $<BUILD_INTERFACE:${RapidJSON_SOURCE_DIR}/include>)
elseif(RapidJSON_INCLUDE_DIRS)
  target_include_directories(NeuralTextures PUBLIC $<BUILD_INTERFACE:${RapidJSON_INCLUDE_DIRS}>)
elseif(RapidJSON_INCLUDE_DIR)
  target_include_directories(NeuralTextures PUBLIC $<BUILD_INTERFACE:${RapidJSON_INCLUDE_DIR}>)
elseif(RapidJSON_DIR)
  # RapidJSON_DIR points to build dir, source dir is typically ../rapidjson-src/include
  get_filename_component(RAPIDJSON_PARENT_DIR "${RapidJSON_DIR}" DIRECTORY)
  if(EXISTS "${RAPIDJSON_PARENT_DIR}/rapidjson-src/include")
    target_include_directories(NeuralTextures PUBLIC $<BUILD_INTERFACE:"${RAPIDJSON_PARENT_DIR}/rapidjson-src/include">)
  endif()
else()
  # Fallback: try to find it
  find_path(RAPIDJSON_INCLUDE_DIR rapidjson/document.h)
  if(RAPIDJSON_INCLUDE_DIR)
    target_include_directories(NeuralTextures PUBLIC $<BUILD_INTERFACE:${RAPIDJSON_INCLUDE_DIR}>)
  endif()
endif()

set_target_properties(NeuralTextures PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(NeuralTextures PRIVATE -Wno-class-memaccess)
endif()

install(TARGETS NeuralTextures
  EXPORT NeuralTexturesTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/OptiXToolkit/NeuralTextures
  )

install(EXPORT NeuralTexturesTargets
  FILE NeuralTexturesTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::
  )

if( BUILD_TESTING )
  add_subdirectory( tests )
endif()

