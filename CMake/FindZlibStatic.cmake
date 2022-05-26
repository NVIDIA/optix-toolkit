#
#  Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

# This is a wrapper for FindZLIB that returns the static library instead of the DSO/DLL.

# Optional input variable: ZlibStatic_ROOT
# Output variables:
#   ZlibStatic_FOUND
#   ZlibStatic_INCLUDE_DIR
#   ZlibStatic_LIBRARIES
#   ZlibStatic_VERSION

# FindZLIB honors ZLIB_ROOT, but the lack of a pre-existing cache entry for it is not user-friendly.
set( ZlibStatic_ROOT "" CACHE PATH "Path to Zlib installation directory" )
if( ZlibStatic_ROOT AND NOT ZLIB_ROOT )
  set( ZLIB_ROOT "${ZlibStatic_ROOT}" CACHE PATH "Path to Zlib installation directory" FORCE )
  unset( ZLIB_INCLUDE_DIR CACHE )
  unset( ZLIB_LIBRARY_RELEASE CACHE )
  unset( ZLIB_LIBRARY_DEBUG CACHE )
endif()

find_package( ZLIB )
if( NOT ZLIB_FOUND OR NOT ZLIB_LIBRARY_RELEASE )
  return()
endif()

# Verify that zlibstatic exists alongside the zlib library.
get_filename_component( LIB_DIR ${ZLIB_LIBRARY_RELEASE} DIRECTORY )

get_filename_component( LIB_FILE_RELEASE ${ZLIB_LIBRARY_RELEASE} NAME )
string( REGEX REPLACE "zlib" "zlibstatic" LIB_FILE_RELEASE "${LIB_FILE_RELEASE}" )
file( GLOB ZlibStatic_LIBRARY_RELEASE "${LIB_DIR}/${LIB_FILE_RELEASE}" )

if( ZLIB_LIBRARY_DEBUG )
  get_filename_component( LIB_FILE_DEBUG ${ZLIB_LIBRARY_DEBUG} NAME )
  string( REGEX REPLACE "zlib" "zlibstatic" LIB_FILE_DEBUG "${LIB_FILE_DEBUG}" )
  file( GLOB ZlibStatic_LIBRARY_DEBUG "${LIB_DIR}/${LIB_FILE_DEBUG}" )
else()
  # Fall back on release library if debug library is not found.
  set( ZlibStatic_LIBRARY_DEBUG "${ZlibStatic_LIBRARY_RELEASE}"
    CACHE FILEPATH "Path to debug Zlib library" )
endif()

if ( ZlibStatic_LIBRARY_RELEASE AND ZlibStatic_LIBRARY_DEBUG )
  set( ZlibStatic_LIBRARIES "optimized;${ZlibStatic_LIBRARY_RELEASE};debug;${ZlibStatic_LIBRARY_DEBUG}"
    CACHE STRING "Zlib static libraries" )
endif()
set( ZlibStatic_INCLUDE_DIR "${ZLIB_INCLUDE_DIR}"
  CACHE PATH "Path to Zlib include directory" )
set( ZlibStatic_VERSION "${ZLIB_VERSION_STRING}"
  CACHE STRING "Zlib version number" )

find_package_handle_standard_args( ZlibStatic
  REQUIRED_VARS
  ZlibStatic_LIBRARY_RELEASE
  ZlibStatic_INCLUDE_DIR
  VERSION_VAR ZlibStatic_VERSION )

if( ZlibStatic_FOUND )
    add_library( Zlib::Static STATIC IMPORTED )
    set_target_properties( Zlib::Static PROPERTIES 
        # Use the release configuration by default
        IMPORTED_LOCATION ${ZlibStatic_LIBRARY_RELEASE}
        IMPORTED_LOCATION_DEBUG ${ZlibStatic_LIBRARY_DEBUG} )
    set_property( TARGET Zlib::Static APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ZLIB_INCLUDE_DIR} )
endif()
