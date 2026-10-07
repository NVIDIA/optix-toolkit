# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

if(TARGET mdl::mdl_sdk)
  return()
endif()

if(OTK_USE_VCPKG OR NOT OTK_FETCH_CONTENT)
  find_package(mdl CONFIG REQUIRED)
  return()
endif()

include(FetchContent)

message(VERBOSE "Finding Boost headers for MDL...")
FetchContent_Declare(
  mdl_boost
  URL https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.bz2
  URL_HASH SHA256=af57be25cb4c4f4b413ed692fe378affb4352ea50fbe294a11ef548f4d527d89
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_GetProperties(mdl_boost)
if(NOT mdl_boost_POPULATED)
  cmake_policy(PUSH)
  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()
  FetchContent_Populate(mdl_boost)
  cmake_policy(POP)
endif()

set(MDL_BOOST_INCLUDE_DIR "${mdl_boost_SOURCE_DIR}")
set(_mdl_boost_config_dir "${CMAKE_CURRENT_BINARY_DIR}/mdl-boost-config")
file(MAKE_DIRECTORY "${_mdl_boost_config_dir}")
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/FetchMdlBoostConfig.cmake.in"
  "${_mdl_boost_config_dir}/BoostConfig.cmake"
  @ONLY
)
set(Boost_DIR "${_mdl_boost_config_dir}")

if(NOT clang_PATH)
  set(_mdl_llvm_version 12.0.1)
  set(_mdl_llvm_base_url
    "https://github.com/llvm/llvm-project/releases/download/llvmorg-${_mdl_llvm_version}"
  )

  if(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$")
    set(_mdl_llvm_filename "LLVM-${_mdl_llvm_version}-win64.exe")
    set(_mdl_llvm_hash
      733bfb425af2e7e4f187fca6d9cfdf7ecc9aa846ef2c227d57fad7cc67d114bde27e49385df362cb399c4aa0e2d481890e2148756a18925b0229ad516a9f8bb4
    )
    set(_mdl_llvm_download_options
      DOWNLOAD_NO_EXTRACT TRUE
      PATCH_COMMAND "<DOWNLOADED_FILE>" /S "/D=<SOURCE_DIR>"
    )
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux"
      AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$")
    set(_mdl_llvm_filename
      "clang+llvm-${_mdl_llvm_version}-x86_64-linux-gnu-ubuntu-16.04.tar.xz"
    )
    set(_mdl_llvm_hash
      6f1eb4ef9885ea7ce56581000e42595f72be37901c213377c8716d160b84441fd017a0a062b188e574a6873b320d3bf2c850beb9822cf4c0025c543effb37a00
    )
  else()
    message(FATAL_ERROR
      "Pre-built Clang binaries required by MDL are not available for "
      "${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}. Set clang_PATH."
    )
  endif()

  message(VERBOSE "Finding Clang for MDL...")
  FetchContent_Declare(
    mdl_clang
    URL "${_mdl_llvm_base_url}/${_mdl_llvm_filename}"
    URL_HASH "SHA512=${_mdl_llvm_hash}"
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    ${_mdl_llvm_download_options}
  )
  FetchContent_MakeAvailable(mdl_clang)
  set(clang_PATH
    "${mdl_clang_SOURCE_DIR}/bin/clang${CMAKE_EXECUTABLE_SUFFIX}"
    CACHE FILEPATH "Path of the Clang binary." FORCE
  )
endif()

if(NOT EXISTS "${clang_PATH}")
  message(FATAL_ERROR "Clang executable not found: ${clang_PATH}")
endif()

find_program(python_PATH NAMES python3 python REQUIRED)

message(VERBOSE "Finding MDL SDK...")
FetchContent_Declare(
  mdl
  URL https://github.com/NVIDIA/MDL-SDK/archive/2024.1.tar.gz
  URL_HASH SHA512=879566a5d70d181d090ae896e585a977ea5048c3e70c9da7058a8ce1b286132dab4e7ad6ccdb019a4d6e43dcc4195659cfbca504d93dbd0f58407001563b5315
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_GetProperties(mdl)
if(NOT mdl_POPULATED)
  cmake_policy(PUSH)
  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()
  FetchContent_Populate(mdl)
  cmake_policy(POP)
endif()

set(MDL_BASE_FOLDER "${mdl_SOURCE_DIR}" CACHE PATH "MDL source directory" FORCE)
set(MDL_INCLUDE_FOLDER "${mdl_SOURCE_DIR}/include" CACHE PATH
  "MDL include directory" FORCE)
set(MDL_SRC_FOLDER "${mdl_SOURCE_DIR}/src" CACHE PATH
  "MDL implementation directory" FORCE)
set(MDL_EXAMPLES_FOLDER "${mdl_SOURCE_DIR}/examples" CACHE PATH
  "MDL examples directory" FORCE)
set(MDL_DOC_FOLDER "${mdl_SOURCE_DIR}/doc" CACHE PATH
  "MDL documentation directory" FORCE)

set(MDL_BUILD_SDK ON CACHE BOOL "Build the MDL SDK" FORCE)
set(MDL_BUILD_SDK_EXAMPLES OFF CACHE BOOL "Build MDL SDK examples" FORCE)
set(MDL_BUILD_CORE_EXAMPLES OFF CACHE BOOL "Build MDL Core examples" FORCE)
set(MDL_BUILD_DOCUMENTATION OFF CACHE BOOL "Build MDL documentation" FORCE)
set(MDL_BUILD_ARNOLD_PLUGIN OFF CACHE BOOL "Build the MDL Arnold plugin" FORCE)
set(MDL_BUILD_DDS_PLUGIN OFF CACHE BOOL "Build the MDL DDS plugin" FORCE)
set(MDL_BUILD_OPENIMAGEIO_PLUGIN OFF CACHE BOOL
  "Build the MDL OpenImageIO plugin" FORCE)
set(MDL_ENABLE_UNIT_TESTS OFF CACHE BOOL "Build MDL unit tests" FORCE)
set(MDL_ENABLE_PYTHON_BINDINGS OFF CACHE BOOL "Build MDL Python bindings" FORCE)
set(MDL_TREAT_RUNTIME_DEPS_AS_BUILD_DEPS OFF CACHE BOOL
  "Treat MDL runtime dependencies as build dependencies" FORCE)

function(_otk_add_mdl_subdirectory)
  # MDL requires a toolchain file for standalone Windows builds. OTK has
  # already initialized its toolchain, so provide an inert file to satisfy
  # that check when MDL is embedded as a subdirectory.
  if(WIN32 AND NOT CMAKE_TOOLCHAIN_FILE)
    set(_mdl_toolchain "${mdl_BINARY_DIR}/otk-empty-toolchain.cmake")
    file(WRITE "${_mdl_toolchain}" "# Intentionally empty.\n")
    set(CMAKE_TOOLCHAIN_FILE "${_mdl_toolchain}")
  endif()
  add_subdirectory("${mdl_SOURCE_DIR}" "${mdl_BINARY_DIR}" EXCLUDE_FROM_ALL)
endfunction()

_otk_add_mdl_subdirectory()
