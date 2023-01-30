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

option( OTK_FETCH_CONTENT "Use FetchContent for third party libraries" ON )
if( NOT OTK_FETCH_CONTENT )
  find_package( OpenEXR 3.1 REQUIRED )
else()
  include(FetchContent)

  FetchContent_Declare(
    Imath
    GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/Imath.git
    GIT_TAG v3.1.5
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(Imath)

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

  FetchContent_Declare(
    OpenEXR
    GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openexr.git
    GIT_TAG v3.1.5
    GIT_SHALLOW TRUE
    PATCH_COMMAND git restore . && git apply --ignore-space-change --ignore-whitespace ${CMAKE_CURRENT_LIST_DIR}/OpenEXRSetup.cmake.patch
  )
  FetchContent_MakeAvailable(OpenEXR)
endif()

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
