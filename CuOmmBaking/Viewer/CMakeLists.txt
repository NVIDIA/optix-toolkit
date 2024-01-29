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
include(FetchGlfw)

include(embed_cuda)
embed_cuda(
  CONST HEADER CuOmmBakingViewerKernelCuda.h
  OUTPUT_TARGET
    ommBakingViewerKernel
  FOLDER
    Examples/CmOmmBaking
  LIBRARIES
    OptiXToolkit::CuOmmBaking 
    OptiXToolkit::ShaderUtil
  SOURCES
    CuOmmBakingViewer.cu
)

otk_add_executable( ommBakingViewer
  CuOmmBakingViewer.cpp
  CuOmmBakingApp.cu
  CuOmmBakingApp.cpp
  CuOmmBakingApp.h
  CuOmmBakingAppDeviceUtil.h
  LaunchParams.h
  PerDeviceOptixState.h
  Procedural.h
  Util/interval_math.h
  SourceDir.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/SourceDir.h
  )

source_group( "Header Files\\Util" FILES
  Util/interval_math.h
)

target_include_directories(ommBakingViewer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>  # for SourceDir.h
  )

# Embed CMAKE_CURRENT_SOURCE_DIR in getSourceDir()
configure_file( SourceDir.h.in include/SourceDir.h @ONLY )

target_link_libraries( ommBakingViewer
  PUBLIC  
  OptiXToolkit::CuOmmBaking
  OptiXToolkit::Gui
  OptiXToolkit::ShaderUtil
  OptiXToolkit::Util
  OptiX::OptiX
  PRIVATE
  ommBakingViewerKernel
  glfw
  glad::glad
  )

set_target_properties( ommBakingViewer PROPERTIES
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit
  FOLDER Examples/CmOmmBaking )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:ommBakingViewer> )
  add_custom_command( TARGET ommBakingViewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:ommBakingViewer> $<TARGET_FILE_DIR:ommBakingViewer>
    COMMAND_EXPAND_LISTS )
endif()
