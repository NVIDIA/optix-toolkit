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

if(TARGET GTest::gtest)
  return()
endif()

if(OTK_USE_VCPKG)
    find_package(GTest CONFIG REQUIRED)
    return()
endif()

function(gtest_folders)
  foreach(_target GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
    get_target_property(_alias ${_target} ALIASED_TARGET)
    if(_alias)
      set_property(TARGET ${_alias} PROPERTY FOLDER ThirdParty/gtest)
    endif()
  endforeach()
endfunction()

if( NOT OTK_FETCH_CONTENT )
  find_package( GTest REQUIRED )
  gtest_folders()
  return()
endif()

include(FetchContent)

set( INSTALL_GTEST OFF CACHE BOOL "Enable installation of googletest" )

message(VERBOSE "Finding googletest...")
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.11.0
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS NAMES GTest
)
FetchContent_MakeAvailable(googletest)

gtest_folders()

# Without this interface definition, clients of gmock will get this link error:
#   unresolved external symbol "class testing::internal::Mutex testing::internal::g_gmock_mutex"
#   unresolved external symbol "class testing::internal::ThreadLocal<class testing::Sequence *> testing::internal::g_gmock_implicit_sequence"
get_target_property(_gmock_alias GTest::gmock ALIASED_TARGET)
get_target_property(_gmock_main_alias GTest::gmock_main ALIASED_TARGET)
if(BUILD_SHARED_LIBS)
    if(_gmock_alias)
      target_compile_definitions( ${_gmock_alias} INTERFACE GTEST_LINKED_AS_SHARED_LIBRARY )
    endif()
    if(_gmock_main_alias)
      target_compile_definitions( ${_gmock_main_alias} INTERFACE GTEST_LINKED_AS_SHARED_LIBRARY )
    endif()
endif()
