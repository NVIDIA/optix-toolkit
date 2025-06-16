# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

#########################################################
# Welcome to the OptiX Toolkit (OTK)

# If you have any questions, we encourage you to post on the OptiX forums:
# https://devtalk.nvidia.com/default/board/90/

# CMake helper files are located in the CMake subdirectory.
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake )

# We rely on CMake 3.24 for its integration of FetchContent and find_package.
# We rely on CMake 3.27 for proper CUDA compilation to optixir.
# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required( VERSION 3.27 FATAL_ERROR )
include( BuildConfig )
include( Policies )
include( ProjectOptions )
include( SetCxxStandard )

#########################################################
# Set the project name (i.e. the VS solution file).
project( ${OTK_PROJECT_NAME} LANGUAGES C CXX CUDA VERSION 0.9.1 )
set( OTK_PROJECT_SOURCE_DIR ${${OTK_PROJECT_NAME}_SOURCE_DIR} )
set( OTK_PROJECT_BINARY_DIR ${${OTK_PROJECT_NAME}_BINARY_DIR} )
set_cxx_standard( 11 )
set( CMAKE_CXX_EXTENSIONS OFF )

#########################################################
# Enable testing with CTest.
if( OTK_BUILD_TESTS )
    include( CTest )
else()
    set( BUILD_TESTING OFF )
endif()

#########################################################
# Create and install package configuration and version files.

include( GNUInstallDirs )

configure_file( ${OTK_PROJECT_SOURCE_DIR}/CMake/OptiXToolkitConfig.cmake.in ${OTK_PROJECT_BINARY_DIR}/CMake/OptiXToolkitConfig.cmake @ONLY )
configure_file( ${OTK_PROJECT_SOURCE_DIR}/CMake/OptiXToolkitConfigVersion.cmake.in ${OTK_PROJECT_BINARY_DIR}/CMake/OptiXToolkitConfigVersion.cmake @ONLY )

install( FILES ${OTK_PROJECT_BINARY_DIR}//CMake/OptiXToolkitConfig.cmake
  ${OTK_PROJECT_BINARY_DIR}/CMake/OptiXToolkitConfigVersion.cmake
  CMake/FindOptiX.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit )

#########################################################
# Third-party libraries

find_package( CUDAToolkit )
include(FetchOptiX)

#########################################################
# Subdirectories

function( checkSubDirectory subdir )
    if( NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/${subdir}/CMakeLists.txt )
        message( STATUS "Requested libraries: ${OTK_LIBRARIES}" )
        message( STATUS "Build PyOptiX: ${OTK_BUILD_PYOPTIX}" )
        message( STATUS "Build examples: ${OTK_BUILD_EXAMPLES}" )
        message( STATUS "Build docs: ${OTK_BUILD_DOCS}" )
        message( FATAL_ERROR "Missing required subdirectory ${subdir}" )
    endif()
    add_subdirectory( ${subdir} )
endfunction()

if( NOT OTK_LIBRARIES OR OTK_LIBRARIES STREQUAL "ALL" )
    set( OTK_LIBRARIES "DemandLoading;Memory;OmmBaking;ShaderUtil" )
endif()
set( DemandLoading_DEPENDENCIES "Memory;ShaderUtil" )
set(        Memory_DEPENDENCIES "" )
set(     OmmBaking_DEPENDENCIES "Memory" )
set(    ShaderUtil_DEPENDENCIES "Memory" )
set( buildSubDirs "" )
foreach( lib ${OTK_LIBRARIES} )
    if( NOT ${lib} IN_LIST buildSubDirs )
        list( APPEND buildSubDirs ${lib} ${${lib}_DEPENDENCIES} )
    endif()
endforeach()
# The examples depend on ShaderUtil
if( OTK_BUILD_EXAMPLES )
    list( APPEND buildSubDirs "ShaderUtil" )
endif()
list( REMOVE_DUPLICATES buildSubDirs )
list( SORT buildSubDirs )
foreach( subdir ${buildSubDirs} )
    checkSubDirectory( ${subdir} )
endforeach()

# Handle these subdirectories based on their specific options.
if( OTK_BUILD_PYOPTIX )
  checkSubDirectory( PyOptiX )
endif()

if( OTK_BUILD_EXAMPLES )
   checkSubDirectory( examples )
endif()

if( OTK_BUILD_DOCS )
  checkSubDirectory( docs/API )
endif()
