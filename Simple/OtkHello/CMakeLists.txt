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

include(GNUInstallDirs)
include(BuildConfig)
include(FetchGlad)
include(embed_ptx)

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 11.0 REQUIRED )
endif()

embed_ptx(
  CONST HEADER otkHelloKernelPTX.h
  OUTPUT_TARGET
    otkHelloKernel
  FOLDER
    Examples/Simple
  PTX_LINK_LIBRARIES
    OptiXToolkit::ShaderUtil
    OptiX::OptiX
  SOURCES
    draw_solid_color.cu
)

otk_add_executable( otkHello
  hello.cpp
  hello.h
  )

target_link_libraries( otkHello
  otkHelloKernel
  OptiXToolkit::Gui
  OptiXToolkit::Util
  OptiX::OptiX
  CUDA::cuda_driver
  CUDA::cudart
  ${CMAKE_DL_LIBS}  
  )

set_target_properties( otkHello PROPERTIES
  FOLDER Examples/Simple
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:otkHello> )
  add_custom_command( TARGET otkHello POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:otkHello> $<TARGET_FILE_DIR:otkHello>
    COMMAND_EXPAND_LISTS )
endif()
