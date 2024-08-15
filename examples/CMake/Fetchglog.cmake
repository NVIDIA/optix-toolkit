# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

if( TARGET glog )
    return()
endif()

if( NOT OTK_FETCH_CONTENT )
  find_package( glog REQUIRED )
  return()
endif()

include(FetchContent)

message(VERBOSE "Finding glog...")
FetchContent_Declare(
  glog
  GIT_REPOSITORY https://github.com/google/glog.git
  GIT_TAG a6a166db069520dbbd653c97c2e5b12e08a8bb26 # v0.3.5
  GIT_PROGRESS TRUE
  PATCH_COMMAND ${CMAKE_COMMAND} -DGITCOMMAND:PATH=${GITCOMMAND} -DPATCHFILE:PATH=${CMAKE_CURRENT_LIST_DIR}/glog-build-parameters.patch.txt -P ${CMAKE_CURRENT_LIST_DIR}/GitPatch.cmake
)
FetchContent_MakeAvailable(glog)
# Let find_package know we have it
set(glog_FOUND ON PARENT_SCOPE)

set_target_properties(glog PROPERTIES FOLDER "ThirdParty")
