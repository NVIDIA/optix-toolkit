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
target_include_directories( DemandGeometry PUBLIC include )
target_link_libraries( DemandGeometry PUBLIC DemandLoading OptiXToolkit::Error CUDA::cuda_driver )
set_target_properties( DemandGeometry PROPERTIES FOLDER DemandLoading )

add_library(OptiXToolkit::DemandGeometry ALIAS DemandGeometry)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
