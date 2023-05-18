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

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 10.0 REQUIRED )
endif()

include(BuildConfig)
include(CTest)
include(FetchOpenEXR)

add_library(Util STATIC
  include/OptiXToolkit/Util/Exception.h
  include/OptiXToolkit/Util/EXRInputFile.h
  include/OptiXToolkit/Util/Fill.h
  include/OptiXToolkit/Util/Logger.h
  src/EXRInputFile.cpp
  src/Exception.cpp
  src/Logger.cpp
  )

add_library(OptiXToolkit::Util ALIAS Util)

target_include_directories(Util PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OptiX_INCLUDE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_link_libraries(Util PRIVATE
  OptiX::OptiX
  CUDA::cuda_driver
  CUDA::cudart
  OpenEXR::OpenEXR
  )
target_link_libraries(Util PUBLIC OptiXToolkit::Memory)

set_target_properties(Util PROPERTIES FOLDER Examples)

if( BUILD_TESTING )
  add_subdirectory( tests )
endif()
