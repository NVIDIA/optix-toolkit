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
target_include_directories( DemandMaterial PUBLIC include )
target_link_libraries( DemandMaterial PUBLIC DemandLoading OptiXToolkit::Error CUDA::cuda_driver )
set_target_properties( DemandMaterial PROPERTIES FOLDER DemandLoading )

add_library(OptiXToolkit::DemandMaterial ALIAS DemandMaterial)

if( BUILD_TESTING )
    add_subdirectory( tests )
endif()
