#
#  Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

# Find OpenEXR package.

# Optional input variable: OpenEXR_ROOT
# Output variables:
#   OpenEXR_FOUND
#   OpenEXR_IMATH_INCLUDE_DIR (3.1+)
#   OpenEXR_INCLUDE_DIR
#   OpenEXR_LIBRARIES
#   OpenEXR_VERSION

if( OpenEXR_FOUND )
  return()
endif()

# Find OpenEXR includes.
find_path( OpenEXR_INCLUDE_DIR OpenEXRConfig.h
  PATHS "${OpenEXR_ROOT}/include/OpenEXR" NO_DEFAULT_PATH )
if( NOT OpenEXR_INCLUDE_DIR )
  find_path( OpenEXR_INCLUDE_DIR ImfOutputFile.h
    HINTS "${OpenEXR_ROOT}/include/OpenEXR"
    PATH_SUFFIXES OpenEXR
    )
endif()
mark_as_advanced( OpenEXR_INCLUDE_DIR )

# We cannot proceed if the headers are not found
if( NOT OpenEXR_INCLUDE_DIR )
  return()
endif()

# Get version number from header, which we need for the library names.
set( OpenEXR_VERSION "" CACHE STRING "OpenEXR version string" )
set( CONFIG_H "${OpenEXR_INCLUDE_DIR}/OpenEXRConfig.h" )
if( NOT OpenEXR_VERSION AND EXISTS "${CONFIG_H}" )
  message( "Reading OpenEXR version from ${CONFIG_H}" )
  file( STRINGS "${CONFIG_H}" VERSION_STRING
    REGEX "#define OPENEXR_VERSION_STRING" )
  string( REGEX REPLACE ".*\"([0-9.]+)\".*" "\\1" VERSION_STRING "${VERSION_STRING}" )
  set( OpenEXR_VERSION "${VERSION_STRING}" CACHE STRING "OpenEXR version string" FORCE )
endif()
string( REGEX REPLACE "^([0-9]+).*" "\\1" VERSION_MAJOR "${OpenEXR_VERSION}" )
string( REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR "${OpenEXR_VERSION}" )
set( VERSION_SUFFIX "${VERSION_MAJOR}_${VERSION_MINOR}" )

if( ${OpenEXR_VERSION} VERSION_GREATER_EQUAL "3.1.0" )
  set( OpenEXR_LIB_NAMES OpenEXR OpenEXRCore Iex Imath IlmThread )
else()
  set( OpenEXR_LIB_NAMES IlmImf Half Iex Imath IlmThread )
endif()

if( ${OpenEXR_VERSION} VERSION_GREATER_EQUAL "3.1.0" )
  # Find Imath includes (OpenEXR 3.1+).
  find_path( OpenEXR_IMATH_INCLUDE_DIR ImathMath.h
    PATHS "${OpenEXR_ROOT}/include/Imath" NO_DEFAULT_PATH )
  mark_as_advanced( OpenEXR_IMATH_INCLUDE_DIR )
endif()

# Allow location of library directory to be overridden.
set( OpenEXR_LIB_DIR "${OpenEXR_ROOT}/lib" CACHE PATH "Path to OpenEXR libraries" )
mark_as_advanced( OpenEXR_LIB_DIR )

# Find OpenEXR libraries.
set( OpenEXR_LIBRARIES "" )
foreach( LIB ${OpenEXR_LIB_NAMES} )
  find_library( OpenEXR_${LIB}_RELEASE
    NAMES "${LIB}-${VERSION_SUFFIX}_s" "${LIB}-${VERSION_SUFFIX}" "${LIB}_s" "${LIB}"
    HINTS "${OpenEXR_LIB_DIR}" )
  mark_as_advanced( OpenEXR_${LIB}_RELEASE )
  if( OpenEXR_${LIB}_RELEASE )
    list( APPEND OpenEXR_LIBRARIES optimized "${OpenEXR_${LIB}_RELEASE}" )
  endif()

  find_library( OpenEXR_${LIB}_DEBUG
    NAMES "${LIB}-${VERSION_SUFFIX}_s_d" "${LIB}-${VERSION_SUFFIX}_d" "${LIB}_s_d" "${LIB}_d"
    HINTS "${OpenEXR_LIB_DIR}" )
  mark_as_advanced( OpenEXR_${LIB}_DEBUG )
  if( OpenEXR_${LIB}_DEBUG )
    list( APPEND OpenEXR_LIBRARIES debug "${OpenEXR_${LIB}_DEBUG}" )
  elseif( OpenEXR_${LIB}_RELEASE )
    # Fallback: use release libraries if no debug libraries were found.
    list( APPEND OpenEXR_LIBRARIES debug "${OpenEXR_${LIB}_RELEASE}" )
  endif()
endforeach( LIB )

include( FindPackageHandleStandardArgs )

# find_package_handle_standard_args reports the value of the first variable
# on success, so make sure this is the actual OpenEXR library
if( ${OpenEXR_VERSION} VERSION_GREATER_EQUAL "3.1.0" )
  find_package_handle_standard_args( OpenEXR
    REQUIRED_VARS
    OpenEXR_OpenEXR_RELEASE OpenEXR_Iex_RELEASE OpenEXR_Imath_RELEASE OpenEXR_IlmThread_RELEASE
    OpenEXR_INCLUDE_DIR OpenEXR_IMATH_INCLUDE_DIR
    VERSION_VAR OpenEXR_VERSION )
else()
  find_package_handle_standard_args( OpenEXR
    REQUIRED_VARS
    OpenEXR_IlmImf_RELEASE OpenEXR_Half_RELEASE OpenEXR_Iex_RELEASE OpenEXR_Imath_RELEASE OpenEXR_IlmThread_RELEASE
    OpenEXR_INCLUDE_DIR
    VERSION_VAR OpenEXR_VERSION )
endif()

foreach( LIB ${OpenEXR_LIB_NAMES} )
  if( OpenEXR_${LIB}_RELEASE )
    set( target OpenEXR::${LIB} )
    # Remap IlmImf to OpenEXR to unify the exported targets between old and new versions
    if( ${LIB} STREQUAL IlmImf )
      set( target OpenEXR::OpenEXR )
    endif()
    add_library( ${target} STATIC IMPORTED )
    if( WIN32 )
      set_target_properties( ${target} PROPERTIES
        IMPORTED_LOCATION_RELEASE ${OpenEXR_${LIB}_RELEASE}
        IMPORTED_LOCATION_DEBUG ${OpenEXR_${LIB}_DEBUG}
        MAP_IMPORTED_CONFIG_MINSIZEREL Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release )
      # We don't have PDB files for debug builds, so ignore
      # LNK4099 PDB 'filename' was not found with 'object/library' or at 'path'; linking object as if no debug info
      set_property( TARGET ${target} APPEND PROPERTY INTERFACE_LINK_OPTIONS $<$<CONFIG:Debug>:/ignore:4099> )
    else()
      if( ${OpenEXR_${LIB}_DEBUG} )
        set_target_properties( ${target} PROPERTIES
          IMPORTED_LOCATION ${OpenEXR_${LIB}_RELEASE}
          IMPORTED_LOCATION_DEBUG ${OpenEXR_${LIB}_DEBUG}
          MAP_IMPORTED_CONFIG_MINSIZEREL ""
          MAP_IMPORTED_CONFIG_RELWITHDEBINFO "" )
      else()
        set_target_properties( ${target} PROPERTIES
          IMPORTED_LOCATION ${OpenEXR_${LIB}_RELEASE}
          MAP_IMPORTED_CONFIG_DEBUG ""
          MAP_IMPORTED_CONFIG_MINSIZEREL ""
          MAP_IMPORTED_CONFIG_RELWITHDEBINFO "" )
      endif()
    endif()
    if( ${OpenEXR_VERSION} VERSION_GREATER_EQUAL "3.1.0" )
      set_property( TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenEXR_INCLUDE_DIR} ${OpenEXR_IMATH_INCLUDE_DIR} )
    else()
      set_property( TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenEXR_INCLUDE_DIR} )
    endif()
  endif()
endforeach()

# Record the library dependencies for OpenEXR on the other OpenEXR libraries
if( OpenEXR_OpenEXR_RELEASE AND ${OpenEXR_VERSION} VERSION_GREATER_EQUAL "3.1.0")
  set_property( TARGET OpenEXR::OpenEXR PROPERTY INTERFACE_LINK_LIBRARIES
    OpenEXR::OpenEXRCore OpenEXR::Iex OpenEXR::Imath OpenEXR::IlmThread )
elseif( OpenEXR_IlmImf_RELEASE AND OpenEXR_IlmThread_RELEASE )
  set_property( TARGET OpenEXR::OpenEXR PROPERTY INTERFACE_LINK_LIBRARIES
    OpenEXR::Half OpenEXR::Iex OpenEXR::Imath OpenEXR::IlmThread )
endif()
