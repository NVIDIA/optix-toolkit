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

include(FetchContent)

FetchContent_Declare(
  Imath
  GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/Imath.git
  GIT_TAG v3.1.5
)
FetchContent_MakeAvailable(Imath)

set( OPENEXR_BUILD_TOOLS OFF CACHE BOOL "Enables building of utility programs" )
set( OPENEXR_INSTALL_EXAMPLES OFF CACHE BOOL "Install OpenEXR examples" )

FetchContent_Declare(
  OpenEXR
  GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openexr.git
  GIT_TAG v3.1.5
)
FetchContent_MakeAvailable(OpenEXR)

add_library( ImageSource STATIC
  ${CMAKE_SOURCE_DIR}/include/ImageSource/CheckerBoardImage.h
  ${CMAKE_SOURCE_DIR}/include/ImageSource/CoreEXRReader.h
  ${CMAKE_SOURCE_DIR}/include/ImageSource/DeviceConstantImage.h 
  ${CMAKE_SOURCE_DIR}/include/ImageSource/DeviceConstantImageParams.h
  ${CMAKE_SOURCE_DIR}/include/ImageSource/DeviceMandelbrotImage.h
  ${CMAKE_SOURCE_DIR}/include/ImageSource/DeviceMandelbrotParams.h
  ${CMAKE_SOURCE_DIR}/include/ImageSource/EXRReader.h
  ${CMAKE_SOURCE_DIR}/include/ImageSource/ImageSource.h
  ${CMAKE_SOURCE_DIR}/include/ImageSource/MultiCheckerImage.h
  ${CMAKE_SOURCE_DIR}/include/ImageSource/TextureInfo.h
  CheckerBoardImage.cpp
  CoreEXRReader.cpp
  DeviceConstantImage.cpp
  DeviceConstantImageKernels.cu
  DeviceMandelbrotImage.cpp
  DeviceMandelbrotImageKernels.cu
  EXRReader.cpp
  Exception.h
  ImageSource.cpp
  Stopwatch.h
  TextureInfo.cpp
  )

target_include_directories( ImageSource PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_link_libraries( ImageSource PUBLIC OpenEXR::OpenEXR OpenEXR::OpenEXRCore )
target_compile_definitions( ImageSource PUBLIC 
  OPTIX_SAMPLE_USE_OPEN_EXR
  OPTIX_SAMPLE_USE_CORE_EXR
  )

set_property(TARGET ImageSource PROPERTY CUDA_ARCHITECTURES "60")
set_property(TARGET ImageSource PROPERTY CUDA_SEPARABLE_COMPILATION ON)
