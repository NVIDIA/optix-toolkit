# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(BuildConfig)

otk_add_library( DemandMaterial STATIC
    include/OptiXToolkit/DemandMaterial/MaterialLoader.h
    include/OptiXToolkit/DemandMaterial/MaterialLoaderImpl.h
    src/DemandMaterial.cpp
    README.md
)
target_link_libraries( DemandMaterial PUBLIC DemandLoading OptiXToolkit::Error CUDA::cuda_driver )
set_target_properties( DemandMaterial PROPERTIES FOLDER DemandLoading )

add_library(OptiXToolkit::DemandMaterial ALIAS DemandMaterial)

target_sources(DemandMaterial
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/DemandMaterial/MaterialLoader.h
)

target_include_directories( DemandMaterial
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
  src
)

install(TARGETS DemandMaterial
  EXPORT DemandMaterialTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT DemandMaterialTargets
  FILE DemandMaterialTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::
)

if( BUILD_TESTING )
    add_subdirectory( tests )
endif()
