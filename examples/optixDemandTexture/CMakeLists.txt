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

#  OptiX kernel is compiled to PTX.
add_library( DemandTexture OBJECT optixDemandTexture.cu )
set_property( TARGET DemandTexture PROPERTY CUDA_PTX_COMPILATION ON )
set_property( TARGET DemandTexture PROPERTY CUDA_ARCHITECTURES "60" )
target_include_directories( DemandTexture PUBLIC
  ${CMAKE_SOURCE_DIR}/DemandLoading/include
  ${OptiX_INCLUDE}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )  
target_link_libraries( DemandTexture PRIVATE OtkCuda )

# The configuration file records the path to the object library.
configure_file( Config.h.in Config.h @ONLY )

add_executable( optixDemandTexture
  optixDemandTexture.cpp
  optixDemandTexture.h
  )

target_include_directories( optixDemandTexture PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}  # for Config.h
  )

target_link_libraries( optixDemandTexture
  DemandLoading
  OtkGui
  OtkUtil
  )

add_dependencies( optixDemandTexture DemandTexture )
set_property( TARGET optixDemandTexture PROPERTY CUDA_ARCHITECTURES "60" )
