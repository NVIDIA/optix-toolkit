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

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
include(../CMake/Policies.cmake)

project(ImageSource LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(../CMake/BuildConfig.cmake)
include(../CMake/FetchOpenEXR.cmake)

find_package( CUDAToolkit 11.1 REQUIRED )

if(PROJECT_IS_TOP_LEVEL)
  find_package(OptiXToolkit REQUIRED)
endif()

add_library( ImageSource SHARED
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

if(NOT PROJECT_IS_TOP_LEVEL)
  add_library(OptiXToolkit::ImageSource ALIAS ImageSource)
endif()

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
  src/ImageSource/Exception.h
  src/ImageSource/Stopwatch.h
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
  CUDA::cudart_static
  ${CMAKE_DL_LIBS}  
  PRIVATE
  OpenEXR_OTK
  )

set_target_properties(ImageSource PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

install(TARGETS ImageSource
  EXPORT ImageSourceTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(EXPORT ImageSourceTargets
  FILE ImageSourceTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::
  )

set( OTK_BUILD_TESTS ON CACHE BOOL "Enable build of OptiXToolkit tests" )
if( ${OTK_BUILD_TESTS} )
  add_subdirectory( tests )
endif()
