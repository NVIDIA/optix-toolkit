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

add_library( ImageSource
  src/CheckerBoardImage.cpp
  src/CoreEXRReader.cpp
  src/DeviceConstantImage.cpp
  src/DeviceConstantImageKernels.cu
  src/DeviceMandelbrotImage.cpp
  src/DeviceMandelbrotImageKernels.cu
  src/EXRReader.cpp
  src/Exception.h
  src/ImageSource.cpp
  src/Stopwatch.h
  src/TextureInfo.cpp
  )
set_property(TARGET ImageSource PROPERTY FOLDER DemandLoading)

add_library(OptiXToolkit::ImageSource ALIAS ImageSource)

target_sources(ImageSource
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/ImageSource/CheckerBoardImage.h
  include/OptiXToolkit/ImageSource/CoreEXRReader.h
  include/OptiXToolkit/ImageSource/DeviceConstantImage.h
  include/OptiXToolkit/ImageSource/DeviceConstantImageParams.h
  include/OptiXToolkit/ImageSource/DeviceMandelbrotImage.h
  include/OptiXToolkit/ImageSource/DeviceMandelbrotParams.h
  include/OptiXToolkit/ImageSource/EXRReader.h
  include/OptiXToolkit/ImageSource/ImageSource.h
  include/OptiXToolkit/ImageSource/MultiCheckerImage.h
  include/OptiXToolkit/ImageSource/TextureInfo.h
)

source_group( "Header Files\\Implementation" FILES
  src/Exception.h
  src/Stopwatch.h
  )

target_compile_definitions( ImageSource PUBLIC
  OPTIX_SAMPLE_USE_CORE_EXR
  OTK_IMF_NAMESPACE=Imf_3_1
  )

target_include_directories( ImageSource
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  PRIVATE
  src
  )

# Bundle OpenEXR libraries to work around export problem.
add_library(OpenEXR_OTK INTERFACE IMPORTED)
target_link_libraries(OpenEXR_OTK INTERFACE OpenEXR::OpenEXR OpenEXR::OpenEXRCore)

target_link_libraries( ImageSource
  PUBLIC
  CUDA::cuda_driver
  ${CMAKE_DL_LIBS}  
  PRIVATE
  OpenEXR_OTK
  )

set_target_properties(ImageSource PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

if( BUILD_TESTING )
  add_subdirectory( tests )
endif()
