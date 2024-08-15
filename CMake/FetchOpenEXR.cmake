# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

# Multiple OpenEXR targets have a compile option (/EHsc) that confuses nvcc.
# We replace it with $<$<COMPILE_LANGUAGE:CXX>:/EHsc>.
function(otk_replace_eshc _package)
  if(TARGET ${_package})
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
          string(FIND "${_options}" "${cxx_flag}" has_cxx_flag)
          if(${has_cxx_flag} EQUAL -1)
            string(REPLACE "/EHsc" ${cxx_flag} _options "${_options}")
            set_target_properties(${_alias} PROPERTIES INTERFACE_COMPILE_OPTIONS "${_options}")
          endif()
        endif()
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
          target_compile_options(${_alias} INTERFACE "-Wno-deprecated-declarations")
        endif()
      endif()
    endforeach()
  endif()
endfunction()

if(TARGET OpenEXR::OpenEXR)
  otk_replace_eshc(OpenEXR::OpenEXR)
  otk_replace_eshc(OpenEXR::OpenEXRCore)          
  return()
endif()

if(OTK_USE_VCPKG)
    find_package(Imath CONFIG REQUIRED)
    find_package(OpenEXR CONFIG REQUIRED)
    otk_replace_eshc(OpenEXR::OpenEXR)
    otk_replace_eshc(OpenEXR::OpenEXRCore)          
    return()
endif()

if( NOT OTK_FETCH_CONTENT )
  find_package( OpenEXR 3.1 REQUIRED )
else()
  include(FetchContent)

  message(VERBOSE "Finding Imath...")
  FetchContent_Declare(
    Imath
    GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/Imath.git
    GIT_TAG v3.1.7
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS 3.1
  )
  FetchContent_MakeAvailable(Imath)
  # Let find_package know we have it
  set(Imath_FOUND ON PARENT_SCOPE)

  # Note: Imath does not permit installation to be disabled.
  # set( IMATH_INSTALL OFF CACHE BOOL "Install Imath" )

  set( OPENEXR_BUILD_EXAMPLES OFF CACHE BOOL "Enables building of utility programs" )
  set( OPENEXR_BUILD_TOOLS OFF CACHE BOOL "Enables building of utility programs" )

  set( OPENEXR_INSTALL OFF CACHE BOOL "Install OpenEXR libraries" )
  set( OPENEXR_INSTALL_EXAMPLES OFF CACHE BOOL "Install OpenEXR examples" )
  set( OPENEXR_INSTALL_TOOLS OFF CACHE BOOL "Install OpenEXR examples" )

  # Note: disabling pkgconfig installation appears to have no effect.
  set( IMATH_INSTALL_PKG_CONFIG OFF CACHE BOOL "Install Imath.pc file" )
  set( OPENEXR_INSTALL_PKG_CONFIG OFF CACHE BOOL "Install OpenEXR.pc file" )

  message(VERBOSE "Finding OpenEXR...")
  FetchContent_Declare(
    OpenEXR
    GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openexr.git
    GIT_TAG v3.1.7
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS 3.1
  )
  FetchContent_MakeAvailable(OpenEXR)
  # Let find_package know we have it
  set(OpenEXR_FOUND ON PARENT_SCOPE)
endif()

foreach(_target OpenEXR OpenEXRCore OpenEXRUtil IlmThread Iex Imath)
  if(TARGET ${_target})
    set_property(TARGET ${_target} PROPERTY FOLDER ThirdParty/OpenEXR)
  endif()
endforeach()
foreach(_target Continuous Experimental Nightly NightlyMemoryCheck)
  if(TARGET ${_target})
    set_property(TARGET ${_target} PROPERTY FOLDER ThirdParty/OpenEXR/CTestDashboardTargets)
  endif()
endforeach()
if(TARGET zlib_external)
  set_property(TARGET zlib_external PROPERTY FOLDER ThirdParty/ZLib)
endif()
