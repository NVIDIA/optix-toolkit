# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(BuildConfig)

# this is set in CMake/Policies.cmake, I don't understand why it has no effect for this file
if(POLICY CMP0167)  # introduced in CMake 3.30
  cmake_policy(SET CMP0167 NEW) # ignore FindBoost and use BoostConfig
endif()

otk_add_library( ImageSource STATIC
  src/CascadeImage.cpp
  src/CheckerBoardImage.cpp
  src/CompressedTextureCacheManager.cpp
  src/DeviceConstantImage.cpp
  src/DeviceConstantImageKernels.cu
  src/DeviceMandelbrotImage.cpp
  src/DeviceMandelbrotImageKernels.cu
  src/DDSImageReader.cpp
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
  include/OptiXToolkit/ImageSource/CompressedTextureCacheManager.h
  include/OptiXToolkit/ImageSource/DDSImageReader.h
  include/OptiXToolkit/ImageSource/DeviceConstantImage.h
  include/OptiXToolkit/ImageSource/DeviceConstantImageParams.h
  include/OptiXToolkit/ImageSource/DeviceMandelbrotImage.h
  include/OptiXToolkit/ImageSource/DeviceMandelbrotParams.h
  include/OptiXToolkit/ImageSource/ImageHelpers.h
  include/OptiXToolkit/ImageSource/ImageSource.h
  include/OptiXToolkit/ImageSource/ImageSourceCache.h
  include/OptiXToolkit/ImageSource/MipMapImageSource.h
  include/OptiXToolkit/ImageSource/MultiCheckerImage.h
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
    # >= 2.5.3: OIIO selects the lock-free OpenEXR Core reader by default (see OIIOReader.cpp).
    # Check the version explicitly rather than via find_package(): OIIO's package uses
    # SameMajorVersion compatibility, so a 2.5.3 version request would reject a 3.x install.
    find_package(OpenImageIO CONFIG QUIET)
    if(NOT OpenImageIO_FOUND OR OpenImageIO_VERSION VERSION_LESS 2.5.3)
        message( FATAL_ERROR
            "OTK_USE_OIIO requires OpenImageIO >= 2.5.3 for lock-free EXR tile reads "
            "(OIIO OpenEXR Core path); found '${OpenImageIO_VERSION}'." )
    endif()

    # Lock-free EXR reads also require OIIO to have been built against OpenEXR >= 3.1.10.
    # Best-effort check (OIIOReader also probes at runtime): in vcpkg builds OIIO links the
    # same OpenEXR, so verify its version here; warn (do not fail) if it is too old/unknown.
    if( NOT DEFINED OpenEXR_VERSION )
        find_package( OpenEXR 3.1.10 CONFIG QUIET )
    endif()
    if( DEFINED OpenEXR_VERSION AND OpenEXR_VERSION VERSION_LESS 3.1.10 )
        message( WARNING
            "OTK_USE_OIIO: linked OpenEXR ${OpenEXR_VERSION} < 3.1.10; OIIO EXR reads may use "
            "the serialized C++ reader rather than the lock-free Core reader." )
    endif()

    set(Boost_NO_WARN_NEW_VERSIONS ON)
    find_package(Boost CONFIG COMPONENTS system filesystem thread REQUIRED)
    target_sources(ImageSource PRIVATE
        src/OIIOReader.cpp
        include/OptiXToolkit/ImageSource/OIIOReader.h
    )
    target_link_libraries(ImageSource
        PUBLIC # TODO: make this private
        OpenImageIO::OpenImageIO
    )
endif()

# Define OTK_USE_OPENEXR and OTK_USE_OIIO in Config.h
configure_file( src/Config.h.in include/Config.h @ONLY )

target_include_directories(ImageSource PUBLIC  # public to facilitate unit testing.
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>  # for Config.h
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

add_subdirectory( Testing )

if( BUILD_TESTING )
  add_subdirectory( tests )
endif()
