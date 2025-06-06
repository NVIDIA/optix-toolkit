# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(FetchOptiX)
include(Fetchrply)

add_library( PbrtSceneLoader STATIC
    include/OptiXToolkit/PbrtSceneLoader/GoogleLogger.h
    include/OptiXToolkit/PbrtSceneLoader/Logger.h
    include/OptiXToolkit/PbrtSceneLoader/MeshReader.h
    include/OptiXToolkit/PbrtSceneLoader/PlyReader.h
    include/OptiXToolkit/PbrtSceneLoader/SceneDescription.h
    include/OptiXToolkit/PbrtSceneLoader/SceneLoader.h
    GoogleLogger.cpp
    PbrtApiImpl.cpp
    PbrtApiImpl.h
    PbrtSceneLoader.cpp
    PlyReader.cpp
    ReadMe.md
)
source_group("Header Files/Internal" PbrtApiImpl.h)
target_link_libraries( PbrtSceneLoader PUBLIC pbrtApi rply::rply )
target_link_libraries( PbrtSceneLoader PRIVATE OptiX::OptiX ShaderUtil Util )
target_include_directories( PbrtSceneLoader PUBLIC include )
set_property( TARGET PbrtSceneLoader PROPERTY FOLDER Examples/Libraries )

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
