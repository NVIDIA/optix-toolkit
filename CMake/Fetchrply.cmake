#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

if( TARGET rply::rply )
    return()
endif()

if(OTK_USE_VCPKG)
    find_package(rply CONFIG REQUIRED)
    return()
endif()

if( NOT OTK_FETCH_CONTENT )
  find_package( rply REQUIRED )
  return()
endif()

include(FetchContent)

message(VERBOSE "Fetching rply...")
FetchContent_Declare(
  rply
  GIT_REPOSITORY https://github.com/diegonehab/rply.git
  GIT_TAG 75639e3f12755ca64559d45a4a0bcffc52e04ef8 # v1.1.4
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(rply)

configure_file(${rply_SOURCE_DIR}/rply.h ${rply_SOURCE_DIR}/include/rply/rply.h COPYONLY)
add_library(rply STATIC
    ${rply_SOURCE_DIR}/rply.h
    ${rply_SOURCE_DIR}/rply.c
    ${rply_SOURCE_DIR}/rplyfile.h
)
target_include_directories(rply PUBLIC ${rply_SOURCE_DIR}/include)
set_target_properties(rply PROPERTIES FOLDER ThirdParty)

if (NOT TARGET rply::rply)
    add_library(rply::rply ALIAS rply)
endif()
