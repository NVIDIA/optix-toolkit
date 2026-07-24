# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(ImageTest)

set(referenceMaterials
    matte
    matte-imagemap
    matte-checkerboard
    plastic
    uber
    mirror
    glass
    metal
    substrate
    translucent
    mix
)

add_custom_target(DemandPbrtScenePbrtReferenceResources)
set_property(TARGET DemandPbrtScenePbrtReferenceResources PROPERTY FOLDER Examples/DemandLoading/Tests)
source_group("Scene Files" REGULAR_EXPRESSION ".*\\.pbrt")
source_group("PBRT Reference Images" REGULAR_EXPRESSION "gold-.*\\.png")
source_group("Texture Images" REGULAR_EXPRESSION "matte-imagemap\\.png")

foreach(material ${referenceMaterials})
    set_property(TARGET DemandPbrtScenePbrtReferenceResources APPEND PROPERTY SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/${material}.pbrt"
        "${CMAKE_CURRENT_SOURCE_DIR}/gold-${material}.png"
    )
endforeach()
set_property(TARGET DemandPbrtScenePbrtReferenceResources APPEND PROPERTY SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/matte-imagemap.png"
)

image_test_get_resource_dir(resourceDir DemandPbrtScene)
if(OTK_USE_MDL)
    add_image_test(DemandPbrtScene matte
        FOLDER Examples/DemandLoading/Tests
        RESOURCE_TARGET DemandPbrtScenePbrtReferenceImageTestResources
        RESOURCES matte.pbrt
        ARGS --bg=0/0/0 --dim=32x32 --warmup=2 ${resourceDir}/matte.pbrt
             --use-mdl-materials --mdl-synchronous-compilation)
    add_image_test(DemandPbrtScene matte-checkerboard
        FOLDER Examples/DemandLoading/Tests
        RESOURCES matte-checkerboard.pbrt
        ARGS --bg=0/0/0 --dim=32x32 --warmup=4 ${resourceDir}/matte-checkerboard.pbrt
             --use-mdl-materials --mdl-synchronous-compilation)
    if(OTK_USE_OIIO)
        add_image_test(DemandPbrtScene matte-imagemap
            FOLDER Examples/DemandLoading/Tests
            RESOURCES matte-imagemap.pbrt matte-imagemap.png
            ARGS --bg=0/0/0 --dim=32x32 --warmup=4 ${resourceDir}/matte-imagemap.pbrt
                 --use-mdl-materials --mdl-synchronous-compilation)
    endif()
endif()
