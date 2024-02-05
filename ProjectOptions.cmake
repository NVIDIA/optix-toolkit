#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#########################################################
# Project options (must precede project() for vcpkg mechanismn to work properly.)

# Default to using vcpkg for dependencies
option( OTK_USE_VCPKG         "Use vcpkg for third party libraries" ON )
option( OTK_USE_VCPKG_OPENEXR "Use vcpkg to obtain OpenEXR" ${OTK_USE_VCPKG} )
option( OTK_USE_OIIO          "Use OpenImageIO to allow DemandLoading to read PNGs and JPGs" OFF )
# OTK_USE_VCPKG takes precedence over FetchContent if both are ON.
option( OTK_FETCH_CONTENT     "Use FetchContent for third party libraries" ON )
option( OTK_BUILD_EXAMPLES    "Enable build of OptiXToolkit examples" ON )
option( OTK_BUILD_TESTS       "Enable build of OptiXToolkit test" ON )
option( OTK_BUILD_DOCS        "Enable build of OptiXToolkit documentation" OFF )
option( OTK_BUILD_PYOPTIX     "Enable build of PyOptiX libraries" ON )

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
    otk_vcpkg_feature( OTK_USE_VCPKG_OPENEXR  "otk-openexr" )
    # OpenImageIO is too costly to include by default (it depends on Boost).
    otk_vcpkg_feature( OTK_USE_OIIO           "otk-openimageio" )
    otk_vcpkg_feature( OTK_BUILD_EXAMPLES     "otk-examples" )
    otk_vcpkg_feature( OTK_BUILD_TESTS        "otk-tests" )

    if( NOT CMAKE_TOOLCHAIN_FILE )
        # Assume that vcpkg submodule is sibling of current list dir
        set( OTK_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake )
        if( NOT EXISTS ${OTK_TOOLCHAIN_FILE} )
            message( FATAL_ERROR "Could not locate vcpkg toolchain file ${OTK_TOOLCHAIN_FILE}" )
        endif()
        include( ${OTK_TOOLCHAIN_FILE} )
    endif()
endif()
