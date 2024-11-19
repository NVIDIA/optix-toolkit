# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

#########################################################
# Project options (must precede project() for vcpkg mechanismn to work properly.)

# Guard against being included multiple times.
if( _OTK_PROJECT_OPTIONS_SET )
    return()
endif()
set( _OTK_PROJECT_OPTIONS_SET ON )

# Default to using vcpkg for dependencies
option( OTK_USE_VCPKG         "Use vcpkg for third party libraries" ON )
option( OTK_USE_OPENEXR       "Use OpenEXR in DemandLoading to read EXRs" ON )
option( OTK_USE_OIIO          "Use OpenImageIO to allow DemandLoading to read PNGs and JPGs" OFF )
# OTK_USE_VCPKG takes precedence over FetchContent if both are ON.
option( OTK_FETCH_CONTENT     "Use FetchContent for third party libraries, if OTK_USE_VCPKG is OFF" ON )
option( OTK_BUILD_EXAMPLES    "Enable build of OptiXToolkit examples" ON )
option( OTK_BUILD_TESTS       "Enable build of OptiXToolkit test" ON )
option( OTK_BUILD_DOCS        "Enable build of OptiXToolkit documentation" ON )
option( OTK_BUILD_PYOPTIX     "Enable build of PyOptiX libraries" OFF )
option( OTK_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF )

# If vcpkg is enabled, ProjectOptions must be included before project() is called.
if( OTK_USE_VCPKG AND PROJECT_NAME )
    message( FATAL_ERROR "Include ProjectOptions before calling project()." )
endif()

set( OTK_PROJECT_NAME   "OptiXToolkit"  CACHE STRING "Project name for use in IDEs (default: OptiXToolkit)" )
set( OTK_LIBRARIES      "ALL"           CACHE STRING "List of libraries to build (default: ALL)" )

# Do some sanity checking on OTK_BUILD_EXAMPLES
if( NOT OTK_LIBRARIES STREQUAL "ALL" AND OTK_BUILD_EXAMPLES )
    set( OTK_LIBRARY_EXAMPLES "DemandLoading;OmmBaking" )
    set( haveExamples FALSE )
    foreach( lib ${OTK_LIBRARY_EXAMPLES} )
        if( ${lib} IN_LIST OTK_LIBRARIES )
            set( haveExamples TRUE )
            break()
        endif()
    endforeach()
    if( NOT haveExamples )
        string(REPLACE ";" ", " printLibs "${OTK_LIBRARIES}")
        message( WARNING "OTK_BUILD_EXAMPLES is ON, but the requested libraries ${printLibs} do not have any examples; forcing OTK_BUILD_EXAMPLES to OFF" )
        set( OTK_BUILD_EXAMPLES OFF CACHE BOOL "Enable build of OptiXToolkit examples" FORCE )
    endif()
endif()

# Needs to be a macro for changes to VCPKG_MANIFEST_FEATURES to be visible in parent scope.
macro( otk_vcpkg_feature var feature )
    if( ${var} AND NOT ${feature} IN_LIST VCPKG_MANIFEST_FEATURES )
        list( APPEND VCPKG_MANIFEST_FEATURES ${feature} )
    endif()
endmacro()

 # Configure vcpkg features for optional dependencies
if( OTK_USE_VCPKG )
    # If not set, start with a well-formed empty list.
    if( NOT VCPKG_MANIFEST_FEATURES )
        set(VCPKG_MANIFEST_FEATURES "" )
    endif()
    otk_vcpkg_feature( OTK_USE_OPENEXR        "otk-openexr" )
    # OpenImageIO is too costly to include by default (it depends on Boost).
    otk_vcpkg_feature( OTK_USE_OIIO           "otk-openimageio" )
    otk_vcpkg_feature( OTK_BUILD_EXAMPLES     "otk-examples" )
    otk_vcpkg_feature( OTK_BUILD_TESTS        "otk-tests" )

    if( NOT CMAKE_TOOLCHAIN_FILE )
        # Assume that vcpkg submodule is sibling of current list dir
        file( REAL_PATH ${CMAKE_CURRENT_LIST_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake OTK_TOOLCHAIN_FILE )
        if( NOT EXISTS ${OTK_TOOLCHAIN_FILE} )
            message( FATAL_ERROR "OTK_USE_VCPKG is ON, but could not locate vcpkg toolchain file ${OTK_TOOLCHAIN_FILE}" )
        endif()
        set( CMAKE_TOOLCHAIN_FILE ${OTK_TOOLCHAIN_FILE} )
    endif()
endif()
