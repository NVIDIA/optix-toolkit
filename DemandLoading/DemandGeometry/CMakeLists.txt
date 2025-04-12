# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(BuildConfig)
find_package( CUDAToolkit 11.1 REQUIRED )

otk_add_library( DemandGeometry STATIC
    include/OptiXToolkit/DemandGeometry/DemandGeometry.h
    include/OptiXToolkit/DemandGeometry/GeometryLoader.h
    include/OptiXToolkit/DemandGeometry/intersectAabb.h
    include/OptiXToolkit/DemandGeometry/ProxyInstances.h
    include/OptiXToolkit/DemandGeometry/ProxyInstancesImpl.h
    src/ProxyInstances.cpp
    README.md
)
target_link_libraries( DemandGeometry PUBLIC DemandLoading OptiXToolkit::Error CUDA::cuda_driver )
set_target_properties( DemandGeometry PROPERTIES FOLDER DemandLoading )

add_library(OptiXToolkit::DemandGeometry ALIAS DemandGeometry)

target_sources(DemandGeometry
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS include
  FILES
  include/OptiXToolkit/DemandGeometry/DemandGeometry.h
  include/OptiXToolkit/DemandGeometry/GeometryLoader.h
  include/OptiXToolkit/DemandGeometry/intersectAabb.h
  include/OptiXToolkit/DemandGeometry/ProxyInstances.h
  include/OptiXToolkit/DemandGeometry/ProxyInstancesImpl.h
)

target_include_directories( DemandGeometry
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
  src
)

install(TARGETS DemandGeometry
  EXPORT DemandGeometryTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT DemandGeometryTargets
  FILE DemandGeometryTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::
)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
