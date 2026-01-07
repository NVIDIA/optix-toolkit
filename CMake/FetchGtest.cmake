# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
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
  GIT_TAG v1.17.0
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS NAMES GTest
)
FetchContent_MakeAvailable(googletest)
# Let find_package know we have it
set(GTest_FOUND ON PARENT_SCOPE)

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
