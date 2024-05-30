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

include(BuildConfig)

otk_add_library( ImageSource
  src/CascadeImage.cpp
  src/CheckerBoardImage.cpp
  src/ImageSource.cpp
  src/ImageSourceCache.cpp
  src/MipMapImageSource.cpp
  src/RateLimitedImageSource.cpp
  src/Stopwatch.h
  src/TextureInfo.cpp
  src/TiledImageSource.cpp
  src/Config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/Config.h
)
source_group("CMake Templates" REGULAR_EXPRESSION ".*\.in$")
set_property(TARGET ImageSource PROPERTY FOLDER DemandLoading)

add_library(OptiXToolkit::ImageSource ALIAS ImageSource)

target_sources(ImageSource
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/ImageSource/CascadeImage.h
  include/OptiXToolkit/ImageSource/CheckerBoardImage.h
  include/OptiXToolkit/ImageSource/ImageHelpers.h
  include/OptiXToolkit/ImageSource/ImageSource.h
  include/OptiXToolkit/ImageSource/ImageSourceCache.h
  include/OptiXToolkit/ImageSource/MipMapImageSource.h
  include/OptiXToolkit/ImageSource/RateLimitedImageSource.h
  include/OptiXToolkit/ImageSource/TextureInfo.h
  include/OptiXToolkit/ImageSource/TiledImageSource.h
  include/OptiXToolkit/ImageSource/WrappedImageSource.h
)

source_group( "Header Files\\Implementation" FILES
  src/Stopwatch.h
  )

target_include_directories( ImageSource
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  PRIVATE
  src
  )

target_link_libraries( ImageSource
  PUBLIC
  CUDA::cuda_driver
  PRIVATE
  OptiXToolkit::Error
  )

if( OTK_USE_OPENEXR )
    include(FetchOpenEXR)
    if(OpenEXR_FOUND)
        target_sources(ImageSource PRIVATE 
            src/CoreEXRReader.cpp
            src/EXRReader.cpp
        )
        target_sources(ImageSource
            PUBLIC 
            FILE_SET HEADERS 
            BASE_DIRS include
            FILES
            include/OptiXToolkit/ImageSource/CoreEXRReader.h
            include/OptiXToolkit/ImageSource/EXRReader.h
        )
        target_compile_definitions( ImageSource PUBLIC
            OPTIX_SAMPLE_USE_CORE_EXR
            OTK_IMF_NAMESPACE=Imf_3_1
        )
        if(NOT MSVC)
            # Silence 'ISO C++ prohibits anonymous structs' warning in OpenEXR header.
            target_compile_options( ImageSource PRIVATE -Wno-pedantic )
        endif()
        # Bundle OpenEXR libraries to work around export problem.
        add_library(OpenEXR_OTK INTERFACE IMPORTED)
        target_link_libraries(OpenEXR_OTK INTERFACE OpenEXR::OpenEXR OpenEXR::OpenEXRCore)
        target_link_libraries( ImageSource
            PRIVATE
            OpenEXR_OTK
        )
    else()
        message( WARNING "OTK_USE_OPENEXR is ON, but OpenEXR not found; skipping EXRReader and forcing to OFF." )
        set( OTK_USE_OPENEXR OFF CACHE BOOL "Use OpenEXR in DemandLoading to read EXRs" FORCE )
    endif()
endif()

if( OTK_USE_OIIO )
    find_package(OpenImageIO QUIET)
    if(OpenImageIO_FOUND)
        set(Boost_NO_WARN_NEW_VERSIONS ON)
        find_package(Boost COMPONENTS system filesystem thread REQUIRED)
        target_sources(ImageSource PRIVATE 
            src/OIIOReader.cpp 
            include/OptiXToolkit/ImageSource/OIIOReader.h
        )
        target_link_libraries(ImageSource
            PUBLIC # TODO: make this private
            OpenImageIO::OpenImageIO
            Boost::filesystem
            Boost::thread
        )
    else()
        message( WARNING "OTK_USE_OIIO is ON, but OpenImageIO not found; skipping OIIOReader and forcing to OFF." )
        set( OTK_USE_OIIO OFF CACHE BOOL "Use OpenImageIO to allow DemandLoading to read PNGs and JPGs" FORCE )
    endif()
endif()

# Define OTK_USE_OPENEXR and OTK_USE_OIIO in Config.h
configure_file( src/Config.h.in include/Config.h @ONLY )

target_include_directories(ImageSource PUBLIC  # public to facilitate unit testing.
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>  # for Config.h
  )

set_target_properties(ImageSource PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

if( BUILD_TESTING )
  add_subdirectory( tests )
endif()
